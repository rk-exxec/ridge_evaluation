from skimage.measure import  CircleModel, ransac, EllipseModel
from skimage.exposure import equalize_adapthist
from sklearn.cluster import KMeans
from skimage import exposure, filters, segmentation
from scipy import stats
from skspatial.objects import Line, Point
import numpy as np
import numba as nb
import diplib as dip
import tifffile
from fractions import Fraction
from pathlib import Path
import cv2
import gzip
import pickle


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

def _ransac_ellipse(points:np.ndarray, n_samples:int=8, n_iterations:int=500, residual_threshold:float=1) -> tuple[CircleModel, np.ndarray]:
    model, inliers = ransac(points, EllipseModel, min_samples=n_samples, max_trials=n_iterations, residual_threshold=residual_threshold)
    return model, inliers  

def _estimate_center(points:np.ndarray):
    # use circle fit to estimate cneter and radius of the rim
    circ, _ = _ransac_circle(points)
    yc, xc, r = (int(round(x)) for x in circ.params)
    residuals = np.sum(circ.residuals(points)**2)
    mean = np.mean(points, axis=0)
    # calculate total sum of squares
    diffs = points - mean
    tss = np.sum((diffs[:,0]+diffs[:,1])**2)

    # calculate r2 score of circle fit
    r2 = 1-residuals/tss
    return yc, xc, r, r2   

def _hough_circle_fit(points:np.ndarray, image:np.ndarray, min_radius:int=0, max_radius:int=0, threshold:int=0) -> tuple[CircleModel, np.ndarray]:
    # use hough circle transform to estimate center and radius of the rim
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius)
    if circles:
        pass
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
    # idk how to preallocate np arrays here, so I use lists
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
def _radial_coord_plane(x_ax, y_ax):
    radial_plane = np.zeros((len(y_ax), len(x_ax)), dtype=np.float32)
    for i in nb.prange(len(x_ax)):
        for j in range(len(y_ax)):
            radial_plane[j,i] = np.sqrt(x_ax[i]**2 + y_ax[j]**2)

    return radial_plane

def intersect_image_borders(image_shape, angles, circle_origin, min_radius) -> dict[float, list[Point]]:
    angle_intersections = dict()
    max_y = image_shape[0]
    max_x = image_shape[1]
    circle_origin = Point(circle_origin)
    rect_lines = [Line.from_points([0,0], [0, max_y]), Line.from_points([0,0], [max_x, 0]), 
                  Line.from_points([0, max_y], [max_x, max_y]), Line.from_points([max_x, 0], [max_x, max_y])]
    
    intersections: list[Point] = list()
    # for all angles, check if the line intersects with the image borders
    # if it does, check if the intesected length includes the expected radius
    # if it does save the angle and the intersection points
    for angle in angles:
        line = Line(circle_origin, direction=[np.cos(angle), np.sin(angle)])
        for rect_line in rect_lines:
            try:
                intersection:Point = line.intersect_line(rect_line)
                # intersection within image borders?
                if 0 <= intersection[0] < max_x and 0 <= intersection[1] < max_y:
                    intersections.append(intersection)
            except ValueError:
                continue


        distances = np.asarray([circle_origin.distance_point(point) for point in intersections])
        if np.any(distances > min_radius):
            min_dist_idx = np.argmin(distances)
            max_dist_idx = np.argmax(distances)
            angle_intersections[angle] = {"first_intersection":np.round(intersections[min_dist_idx]).astype(int), "second_intersection": np.round(intersections[max_dist_idx]).astype(int),
                                          "intersection_distance": intersections[min_dist_idx].distance_point(intersections[max_dist_idx]),
                                          "min_radius":distances[min_dist_idx], "max_radius":distances[max_dist_idx]}

    return angle_intersections


def _round_radial_profile(image:np.ndarray, circ_params:tuple, pixelscale:tuple[float], prune_profiles=True, max_rel_diff_to_circle=0.05) -> tuple[np.ndarray, np.ndarray]:
    from skimage.draw import circle_perimeter
    yc,xc,r = circ_params
    
    circle_fit_radius = r*pixelscale[0]
    

    min_radius = np.min([xc, yc, image.shape[0]-yc, image.shape[1]-xc])
    max_radius = np.max([xc, yc, image.shape[0]-yc, image.shape[1]-xc])
    circles_idx = [circle_perimeter(yc, xc, i, method="andres", shape=image.shape) for i in range(1,max_radius)]
    radii = list()
    for idx in circles_idx:
        z_ax = image[idx]
        radii.append(z_ax)
    # circle_fit_radius = abs(circles_idx[r][1][0]-xc)*pixelscale[0]

    rad = np.zeros(len(radii))
    mean = np.zeros(len(radii))
    stdev = np.zeros(len(radii))
    stderr = np.zeros(len(radii))
    for i,radius in enumerate(radii):
        rad[i] = (1+i)*pixelscale[0]
        z_score = np.abs(stats.zscore(radius))
        radius = radius[z_score<3] # 3 equals 3 sigma
        mean[i] = np.mean(radius)
        stdev[i] = np.std(radius)
        count = len(radius)
        stderr[i] = stdev[i]/np.sqrt(count)
    ridge_rad = radii[r]

    return circle_fit_radius, (rad, mean), stdev, stderr, ridge_rad


# @nb.njit(cache=True, parallel=True)
def _robust_radial_profile(image:np.ndarray, circ_params:tuple, pixelscale:tuple[float], prune_profiles=True, max_rel_diff_to_circle=0.05) -> tuple[np.ndarray, np.ndarray]:
    yc,xc,r = circ_params
    
    # use to get sub-micrometer resolution even when rounding to integers
    accuracy_factor = 1e6

    # work with real dimensions
    # build 3d data array
    ycs,xcs = yc*pixelscale[1]*accuracy_factor, xc*pixelscale[0]*accuracy_factor
    x_values = np.arange(image.shape[1])*pixelscale[0]*accuracy_factor - xcs
    y_values = np.arange(image.shape[0])*pixelscale[1]*accuracy_factor - ycs

    # create a radial coordinate plane for easier calculations
    radial_coord = _radial_coord_plane(x_values, y_values)
    # stack radial coordinate plane with image
    data = np.concatenate((radial_coord[np.newaxis,:,:], image[np.newaxis,:,:]), axis=0)

    # use angles to calculate affected pixels in image, since this is just as effective as using the angular coordinate, but more performant
    # 32 equidistant angles, every angle creates a radius-height profile from the stacked image by using bresenhams algorithm for pixel selection
    
    profiles = list()
    max_y = image.shape[0]
    max_x = image.shape[1]
    if xc < 0 or yc < 0 or xc > max_x or yc > max_y:
        # get new starting positions for the angle profiles to support circular profiles larger than image
        # also remove angles which dont intersect with image
        angle1 = np.arctan2(0-yc, 0-xc)
        angle2 = np.arctan2(0-yc, max_x-xc)
        angle3 = np.arctan2(max_y-yc, 0-xc)
        angle4 = np.arctan2(max_y-yc, max_x-xc)
        angles = np.asarray([angle1, angle2, angle3, angle4])
        # min max comparison doesnt work if 0° is between any of the angles -> rotate 180° if thats the case
        if 0< yc < max_y:
            angles += np.pi
            min_angle = np.min(angles)
            max_angle = np.max(angles)
            angles = np.linspace(min_angle, max_angle, 16)-np.pi
        else:
            min_angle = np.min(angles)
            max_angle = np.max(angles)
            angles = np.linspace(min_angle, max_angle, 16)


        intersection_angles = intersect_image_borders(image.shape, angles, (xc, yc), r+50)
        for angle in intersection_angles.keys():
            start = intersection_angles[angle]["first_intersection"]
            poi = _pixels_along_line(*start, angle, intersection_angles[angle]["intersection_distance"])
            poi = poi[:, (poi[0]>=0) & (poi[0]<image.shape[1]) & (poi[1]>=0) & (poi[1]<image.shape[0])]
            z_ax = data[:, poi[1,:], poi[0,:]]
            profiles.append(z_ax)
        circle_fit_radius = r*np.sqrt(pixelscale[0]**2 + pixelscale[1]**2)*accuracy_factor

    else:
        angles = np.linspace(0, 2*np.pi, 32)
        min_radius = np.min([xc, yc, image.shape[0]-yc, image.shape[1]-xc])
        for angle in angles:
            poi = _pixels_along_line(xc,yc, angle, min_radius)
            poi = poi[:, (poi[0]>=0) & (poi[0]<image.shape[1]) & (poi[1]>=0) & (poi[1]<image.shape[0])]
            z_ax = data[:, poi[1,:], poi[0,:]]
            profiles.append(z_ax)
        # gets the radius of the circle at the 3 o'clock position, which is at the peak of the profile
        circle_fit_radius = radial_coord[yc, xc+r]

    # check if peak is at the same radius for all profiles
    # exclude profiles where the peak is not at the same radius
    # aligning_peak = np.max([p[0,np.argmax(p[1,:])] for p in profiles])
    
    if prune_profiles:
        filtered_profiles = [p for p in profiles if np.isclose(circle_fit_radius, p[0,np.argmax(p[1,:])], rtol=max_rel_diff_to_circle)]
    else:
        filtered_profiles = profiles
    # filtered_profiles = [p for p in profiles if np.isclose(circle_fit_radius, p[0,np.argmax(p[1,:])], rtol=max_rel_diff_to_circle)]
    # for p in profiles:
    #     p[0,:] = p[0,:] + (aligning_peak - p[0,np.argmax(p[1,:])])

    merged_profiles = np.hstack(filtered_profiles)

    # https://stackoverflow.com/a/21242776/9173710 using bincount to create mean for radial profile

    r = np.round(merged_profiles[0,:]).astype(int)
    # tbin = np.bincount(r.ravel(), merged_profiles[1,:].ravel())
    # nr = np.bincount(r.ravel())
    # radial_profile = tbin / nr

    bins = np.arange(r.min(), r.max()+1, dtype=np.float32)

    mean, bin_edges, binnumber = stats.binned_statistic(r.ravel(), merged_profiles[1,:].ravel(), statistic="mean", bins=bins)
    stdev, _ ,_ = stats.binned_statistic(r.ravel(), merged_profiles[1,:].ravel(), statistic="std", bins=bins)
    count, _ ,_ = stats.binned_statistic(r.ravel(), merged_profiles[1,:].ravel(), statistic="count", bins=bins)
    stderr = stdev/np.sqrt(count)



    # using histogram on selected slivces as it supports negative radii
    # r = np.round(merged_profiles[0,:]).astype(int)
    # tbin = np.histogram(r, bins=np.arange(r.min(), r.max()+1), weights=merged_profiles[1,:])[0]
    # nr = np.histogram(r, bins=np.arange(r.min(), r.max()+1))[0]
    # radial_profile = tbin / nr

    # rescale to meters
    bins = bins/accuracy_factor
    return circle_fit_radius/accuracy_factor, (bins[:-1], mean), stdev, stderr

def find_features(image, rim_finder_args):
            
    points, rim = _rim_from_image(image, *rim_finder_args)
    yc, xc, r, r2 = _estimate_center(points)

    return yc, xc, r, r2, rim

def open_surface_img(image_file):
    if isinstance(image_file, str):
        image_file = Path(image_file)
    if image_file.suffix == ".gz":
        with gzip.open(image_file, "rb") as f:
            meatadata, pixelscale, image = pickle.load(f, encoding="latin1")
            pixelscale = pixelscale[0]*1e-6, pixelscale[1]*1e-6
    else:
        with tifffile.TiffFile(image_file) as tif:
            image = tif.asarray()
            ome_metadata = tif.ome_metadata
            metadata = tif.pages[0].tags
            # The TIFF types of the XResolution and YResolution tags are RATIONAL (5)
            # which is defined in the TIFF specification as two longs, the first of which is the numerator and the second the denominator.
            xres = Fraction(*metadata["XResolution"].value)*1e6
            yres = Fraction(*metadata["YResolution"].value)*1e6
            pixelscale = (float(1/xres), float(1/yres))
    return image, pixelscale

        
def extract_ridge_from_image(image_file, robust=True, kmeans_n_clusters=50, threshold=0.5, use_kmeans=True, butterworth_cutoff=0.001, gamma_correction=3, max_rel_diff_to_circle=0.05, prune_profiles=True):
    image, pixelscale = open_surface_img(image_file)
    yc, xc, r, r2, rim = find_features(image, (threshold, use_kmeans, kmeans_n_clusters, butterworth_cutoff, gamma_correction))

    # try:
    if not robust:
        radial_profile = _radial_profile(image, (yc,xc), pixelscale, max_rel_diff_to_circle)
        circle_fit_radius = r*np.sqrt(pixelscale[0]**2 + pixelscale[1]**2)
    else: 
        circle_fit_radius,radial_profile, stdev, sterr = _robust_radial_profile(image, (yc,xc,r), pixelscale, max_rel_diff_to_circle=max_rel_diff_to_circle, prune_profiles=prune_profiles)

    return circle_fit_radius,radial_profile, image, (yc,xc,r), rim, r2, pixelscale, stdev, sterr

def extract_ridge_round(image_file, robust=True, kmeans_n_clusters=50, threshold=0.5, use_kmeans=True, butterworth_cutoff=0.001, gamma_correction=3, max_rel_diff_to_circle=0.05, prune_profiles=True):
    image, pixelscale = open_surface_img(image_file)
    yc, xc, r, r2, rim = find_features(image, (threshold, use_kmeans, kmeans_n_clusters, butterworth_cutoff, gamma_correction))

    # try:
    if not robust:
        radial_profile = _radial_profile(image, (yc,xc), pixelscale, max_rel_diff_to_circle)
        circle_fit_radius = r*np.sqrt(pixelscale[0]**2 + pixelscale[1]**2)
    else: 
        circle_fit_radius,radial_profile, stdev, sterr, ridge_rad = _round_radial_profile(image, (yc,xc,r), pixelscale, max_rel_diff_to_circle=max_rel_diff_to_circle, prune_profiles=prune_profiles)

    return circle_fit_radius,radial_profile, image, (yc,xc,r), rim, r2, pixelscale, stdev, sterr, ridge_rad

def extract_ridge_round_img(image, pixelscale, robust=True, kmeans_n_clusters=50, threshold=0.5, use_kmeans=True, butterworth_cutoff=0.001, gamma_correction=3, max_rel_diff_to_circle=0.05, prune_profiles=True):
    yc, xc, r, r2, rim = find_features(image, (threshold, use_kmeans, kmeans_n_clusters, butterworth_cutoff, gamma_correction))

    # try:
    if not robust:
        radial_profile = _radial_profile(image, (yc,xc), pixelscale, max_rel_diff_to_circle)
        circle_fit_radius = r*np.sqrt(pixelscale[0]**2 + pixelscale[1]**2)
    else: 
        circle_fit_radius,radial_profile, stdev, sterr, ridge_rad = _round_radial_profile(image, (yc,xc,r), pixelscale, max_rel_diff_to_circle=max_rel_diff_to_circle, prune_profiles=prune_profiles)

    return circle_fit_radius,radial_profile, (yc,xc,r), rim, r2, stdev, sterr, ridge_rad
    # except Exception as e:
    #     print(e)
    #     return np.array([range(r), [0]*r]), image, (yc,xc,r), rim, r2, pixelscale, [0]*r, [0]*r

def extract_partial_ridge(image_file, ridge_only_files=[], robust=True, kmeans_n_clusters=50, threshold=0.5, use_kmeans=True, butterworth_cutoff=0.001, gamma_correction=3, max_rel_diff_to_circle=0.05):
    image, pixelscale = open_surface_img(image_file)

    yc, xc, r, r2, rim = find_features(image, (threshold, use_kmeans, kmeans_n_clusters, butterworth_cutoff, gamma_correction))
    circle_fit_radius,radial_profile, stdev, sterr = _round_radial_profile(image, (yc,xc,r), pixelscale, prune_profiles=False, max_rel_diff_to_circle=max_rel_diff_to_circle)

    add_profiles = []
    for file in ridge_only_files:
        rimage, rpixelscale = open_surface_img(file)
        add_profiles.append(_round_radial_profile(rimage, (yc,xc,r), rpixelscale, prune_profiles=False, max_rel_diff_to_circle=max_rel_diff_to_circle) + (float(file.stem.split("_")[-2]),))

    return circle_fit_radius,radial_profile, image, (yc,xc,r), rim, r2, pixelscale, stdev, sterr , add_profiles

