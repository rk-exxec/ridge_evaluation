from skimage.measure import  CircleModel, ransac
from skimage.exposure import equalize_adapthist
from scipy.ndimage import shift
from sklearn.cluster import KMeans
import numpy as np
import diplib as dip
import tifffile
from fractions import Fraction
from pathlib import Path

def _rim_from_image(image:np.ndarray, kmeans_n_clusters=50, threshold=0.5, use_kmeans=True):
    image = equalize_adapthist(image)
    image_normalised = (image - image.min())/(image.max()-image.min())
    if use_kmeans:
        kmeans = KMeans(n_clusters=kmeans_n_clusters, algorithm="lloyd").fit(image_normalised)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        segmented_image = centers[labels]
        segmented_normalized = (segmented_image - segmented_image.min())/(segmented_image.max()-segmented_image.min())
        rim_binarized = segmented_normalized>=threshold
    else:
        rim_binarized = image_normalised>=threshold
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

def _bresenhams_algorithm(origin, angle, radius):
    def plotLine(x0, y0, x1, y1):
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1  else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        error = dx + dy

        line_idxs = list()
        
        while True:
            line_idxs.append((x0, y0))
            if x0 == x1 and y0 == y1: break
            e2 = 2 * error
            if e2 >= dy:
                if x0 >= x1: break
                error = error + dy
                x0 = x0 + sx
            if e2 <= dx:
                if y0 == y1: break
                error = error + dx
                y0 = y0 + sy
        return line_idxs
    
    x0, y0 = origin
    x1 = int(round(x0 + radius * np.cos(angle)))
    y1 = int(round(y0 + radius * np.sin(angle)))
    return plotLine(x0, y0, x1, y1)


def _robust_radial_profile(image:np.ndarray, center:tuple, pixelscale:tuple[float], max_radius:str="inner radius"):
    angles = np.linspace(0, 2*np.pi, 12)
    min_radius = np.min([center[0], center[1], image.shape[0]-center[0], image.shape[1]-center[1]])
    profiles = list()
    # extract profile for 8 angles
    for angle in angles:
        poi = np.asarray(_bresenhams_algorithm(center, angle, min_radius)).reshape(2,-1)
        poi = poi[:, (poi[0]>=0) & (poi[0]<image.shape[1]) & (poi[1]>=0) & (poi[1]<image.shape[0])]

        z_ax = image[poi[1,:], poi[0,:]]
        profiles.append(z_ax)
    # shift profiles so that peaks are aligned
    #furthest_peak_x = np.max([p[0,np.argmax(p[1])] for p in profiles])
    #for p in profiles:
    #    p[0] = p[0] - (furthest_peak_x - p[0,np.argmax(p[1])])
    # align peaks
    furthest_peak_idx = np.max([np.argmax(p) for p in profiles])
    shifted_profiles = list()
    for p in profiles:
        shift =  furthest_peak_idx - np.argmax(p)
        if shift > 0:
            shifted_profiles.append(np.concatenate((np.full(shift, p[0]), p[:-shift])))
        elif shift < 0:
            shifted_profiles.append(np.concatenate((p[-shift:], np.full(-shift, p[-1]))))

    # trim profiles to the shortest length and calculate mean profile
    shortest_length = np.min([len(p) for p in shifted_profiles])
    shifted_profiles = np.array([p[:shortest_length] for p in shifted_profiles])
    z_ax = np.mean(shifted_profiles, axis=0)
    r_ax = np.arange(len(z_ax))*np.mean(pixelscale)

    return np.vstack([r_ax, z_ax])

        
def extract_ridge_from_image(image_file: Path, kmeans_n_clusters=50, threshold=0.5, use_kmeans=True):
    if isinstance(image_file, str):
        image_file = Path(image_file)
    with tifffile.TiffFile(image_file) as tif:
        image_grey = tif.asarray()
        ome_metadata = tif.ome_metadata
        metadata = tif.pages[0].tags
        # The TIFF types of the XResolution and YResolution tags are RATIONAL (5)
        # which is defined in the TIFF specification as two longs, the first of which is the numerator and the second the denominator.
        xres = Fraction(*metadata["XResolution"].value)*1e6
        yres = Fraction(*metadata["YResolution"].value)*1e6
        pixelscale = (1/xres, 1/yres)
            
    points, rim = _rim_from_image(image_grey, kmeans_n_clusters=kmeans_n_clusters, threshold=threshold, use_kmeans=use_kmeans)

    yc, xc, r, r2 = _estimate_center(points)
    print(f"{image_file.stem} Center: ({yc},{xc}) Radius: {r} R2: {r2}")

    # radial_profile = _radial_profile(image_grey, (yc,xc), pixelscale, max_radius="inner radius")
    radial_profile = _robust_radial_profile(image_grey, (yc,xc), pixelscale, max_radius="inner radius")

    return radial_profile, image_grey, (yc,xc,r), rim


