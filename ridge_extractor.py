from skimage.measure import  CircleModel, ransac
from skimage.exposure import equalize_adapthist, equalize_hist
from sklearn.cluster import KMeans
from skimage import exposure, filters, segmentation
import numpy as np
import numba as nb
import diplib as dip
import tifffile
from fractions import Fraction
from pathlib import Path

def _rim_from_image(image:np.ndarray, threshold, use_kmeans, kmeans_n_clusters, butterworth_cutoff, gamma_correction):
    # image = equalize_adapthist(image)
    # image = equalize_hist(image)

    if use_kmeans:
        image = equalize_adapthist(image)
        image_normalised = (image - image.min())/(image.max()-image.min())
        kmeans = KMeans(n_clusters=kmeans_n_clusters, algorithm="lloyd").fit(image_normalised)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        segmented_image = centers[labels]
        segmented_normalized = (segmented_image - segmented_image.min())/(segmented_image.max()-segmented_image.min())
        rim_binarized = segmented_normalized>=threshold
    else:
        image_normalised = (image - image.min())/(image.max()-image.min())
        image_normalised = filters.gaussian(image_normalised, sigma=15)
        image_normalised = exposure.adjust_gamma(image_normalised, gamma_correction)
        image_normalised = filters.butterworth(image_normalised, butterworth_cutoff, high_pass=False)
        image_normalised = exposure.equalize_adapthist(image_normalised)
        # thresh = filters.threshold_otsu(image_normalised)
        # print(thresh)
        rim_binarized = image_normalised>=threshold
        rim_binarized = segmentation.clear_border(rim_binarized)
    points = np.argwhere(rim_binarized)
    return points, rim_binarized

def _ransac_circle(points:np.ndarray, n_samples:int=3, n_iterations:int=500, residual_threshold:float=1) -> tuple[CircleModel, np.ndarray]:
    model, inliers = ransac(points, CircleModel, min_samples=n_samples, max_trials=n_iterations, residual_threshold=residual_threshold)
    return model, inliers   

def _estimate_center(points:np.ndarray):
    circ, _ = _ransac_circle(points)
    yc, xc, r = (int(round(x)) for x in circ.params)
    residuals = np.sum(circ.residuals(points)**2)
    mean = np.mean(points, axis=0)
    # calculate total sum of squares
    diffs = points - mean
    tss = np.sum((diffs[:,0]+diffs[:,1])**2)

    # calculate r2 score
    r2 = 1-residuals/tss
    return yc, xc, r, r2   

def _radial_profile(image:np.ndarray, center:tuple, pixelscale:tuple[float], max_radius:str="inner radius"):
    dipimg = dip.Image(image)
    rad = dip.RadialMean(dipimg, binSize=1, center=center, maxRadius=max_radius)
    prof = np.asarray(rad)
    x_ax = np.arange(len(prof))
    # pixelscale = 0.040639999999999996e-6
    x_ax = x_ax * np.mean(pixelscale)
    return np.vstack((x_ax,prof)) 

@nb.njit(cache=True)
def _pixels_along_line(x1, y1, angle, radius) -> np.ndarray:
    x2 = int(round(x1 + radius * np.cos(angle)))
    y2 = int(round(y1 + radius * np.sin(angle)))    
    # bresenham algorithm
    # from https://de.wikipedia.org/wiki/Bresenham-Algorithmus modified c code
    x,y = x1,y1
    dx = abs(x2 - x1)
    sx = np.sign(x2 - x1)
    dy = abs(y2 -y1)
    sy = np.sign(y2 - y1)

    if dx == 0:
        if sy < 0: 
            start, stop = y2,y1
        else:
            start, stop = y1,y2
        y_coordinates = np.arange(start, stop)
        x_coordinates = x1*np.ones((dy,), dtype=np.int32) 
        return np.vstack((x_coordinates, y_coordinates))

    if dy == 0:
        if sx < 0:
            start, stop = x2,x1
        else:
            start, stop = x1,x2
        x_coordinates = np.arange(start, stop)
        y_coordinates = y1*np.ones((dx,), dtype=np.int32) 
        return np.vstack((x_coordinates, y_coordinates))
    
    if dx > dy:
        pdx, pdy, dfd, dsd = sx, 0, dx, dy
    else:
        pdx, pdy, dfd, dsd = 0, sy, dy, dx

    err = dfd/2
    # Initialize the plotting points
    xcoordinates = list()
    ycoordinates = list()

    xcoordinates.append(x1)
    ycoordinates.append(y1)

    x = x1
    y = y1

    for _ in range(dfd):
        err -= dsd
        if err < 0:
            err += dfd
            x += sx
            y += sy
        else:   
            x += pdx
            y += pdy
        xcoordinates.append(x)
        ycoordinates.append(y)
    return np.vstack((np.asarray(xcoordinates), np.asarray(ycoordinates)))
    
@nb.jit(cache=True, parallel=True)
def radial_coord_plane(x_ax, y_ax):
    radial_plane = np.zeros((len(y_ax), len(x_ax)), dtype=np.float32)
    for i in nb.prange(len(x_ax)):
        for j in nb.prange(len(y_ax)):
            radial_plane[j,i] = np.sqrt(x_ax[i]**2 + y_ax[j]**2)

    return radial_plane

# @nb.njit(cache=True, parallel=True)
def _robust_radial_profile(image:np.ndarray, circ_params:tuple, pixelscale:tuple[float], max_rel_diff_to_circle=0.05) -> tuple[np.ndarray, np.ndarray]:
    yc,xc,r = circ_params
    angles = np.linspace(0, 2*np.pi, 32)

    # work with real dimensions
    # build 3d data array
    # use 1e6 to get micrometer resolution even when rounding to integers
    ycs,xcs = yc*pixelscale[1]*1e6, xc*pixelscale[0]*1e6
    x_values = np.arange(image.shape[1])*pixelscale[0]*1e6 - xcs
    y_values = np.arange(image.shape[0])*pixelscale[1]*1e6 - ycs

    # radial_coord = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    radial_coord = radial_coord_plane(x_values, y_values)

    # x_grid,y_grid = np.meshgrid(x_values, y_values)
    # radial_coord = np.sqrt((x_grid)**2 + (y_grid)**2)
    # angle_coord = np.arctan2(y_grid, x_grid)

    data = np.concatenate((radial_coord[np.newaxis,:,:], image[np.newaxis,:,:]), axis=0)# angle_coord[np.newaxis,:,:] ,

    # use angles to calculate affected pixels in image, since this is just as effective as using the angular coordinate, but more performant
    min_radius = np.min([xc, yc, image.shape[0]-yc, image.shape[1]-xc])
    profiles = list()
    for angle in angles:
        poi = _pixels_along_line(xc, yc, angle, min_radius)
        poi = poi[:, (poi[0]>=0) & (poi[0]<image.shape[1]) & (poi[1]>=0) & (poi[1]<image.shape[0])]
        z_ax = data[:, poi[1,:], poi[0,:]]
        profiles.append(z_ax)

    #shift profiles so that peaks are aligned
    # aligning_peak = np.max([p[0,np.argmax(p[1,:])] for p in profiles])
    circle_fit_radius = radial_coord[yc, xc+r] # gets the radius of the circle at the 3 o'clock position, which is at the peak of the profile
    filtered_profiles = [p for p in profiles if np.isclose(circle_fit_radius, p[0,np.argmax(p[1,:])], rtol=max_rel_diff_to_circle)]
    # for p in profiles:
    #     p[0,:] = p[0,:] + (aligning_peak - p[0,np.argmax(p[1,:])])

    merged_profiles = np.hstack(filtered_profiles)

    # https://stackoverflow.com/a/21242776/9173710 using bincount for radial profile

    r = np.round(merged_profiles[0,:]).astype(int)
    tbin = np.bincount(r.ravel(), merged_profiles[1,:].ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / nr

    # using histogram on selected slivces as it supports negative radii
    # r = np.round(merged_profiles[0,:]).astype(int)
    # tbin = np.histogram(r, bins=np.arange(r.min(), r.max()+1), weights=merged_profiles[1,:])[0]
    # nr = np.histogram(r, bins=np.arange(r.min(), r.max()+1))[0]
    # radial_profile = tbin / nr

    # rescale to meters
    x_ax = np.arange(len(radial_profile), dtype=np.float32)*1e-6
    return x_ax, radial_profile

def find_features(image, rim_finder_args):
            
    points, rim = _rim_from_image(image, *rim_finder_args)
    yc, xc, r, r2 = _estimate_center(points)

    return yc, xc, r, r2, rim

        
def extract_ridge_from_image(image_file, robust=True, kmeans_n_clusters=50, threshold=0.5, use_kmeans=True, butterworth_cutoff=0.001, gamma_correction=3, max_rel_diff_to_circle=0.05):
    if isinstance(image_file, str):
        image_file = Path(image_file)
    with tifffile.TiffFile(image_file) as tif:
        image = tif.asarray()
        ome_metadata = tif.ome_metadata
        metadata = tif.pages[0].tags
        # The TIFF types of the XResolution and YResolution tags are RATIONAL (5)
        # which is defined in the TIFF specification as two longs, the first of which is the numerator and the second the denominator.
        xres = Fraction(*metadata["XResolution"].value)*1e6
        yres = Fraction(*metadata["YResolution"].value)*1e6
        pixelscale = (float(1/xres), float(1/yres))

    yc, xc, r, r2, rim = find_features(image, (threshold, use_kmeans, kmeans_n_clusters, butterworth_cutoff, gamma_correction))

    try:
        if not robust:
            radial_profile = _radial_profile(image, (yc,xc), pixelscale, max_rel_diff_to_circle)
        else: 
            radial_profile = _robust_radial_profile(image, (yc,xc,r), pixelscale, max_rel_diff_to_circle)

        return radial_profile, image, (yc,xc,r), rim, r2, pixelscale
    except Exception as e:
        print(e)
        return np.array([range(r), [0]*r]), image, (yc,xc,r), rim, r2, pixelscale


