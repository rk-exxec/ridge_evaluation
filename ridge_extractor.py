from skimage.measure import EllipseModel, CircleModel
from skimage.exposure import equalize_adapthist, equalize_hist
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
import numpy as np
import diplib as dip
import pandas as pd
import tifffile
from fractions import Fraction
from pathlib import Path

def _rim_from_image(image:np.ndarray, kmeans_n_clusters=50, segmented_threshold=0.5):
    image = equalize_adapthist(image)
    image_normalised = (image - image.min())/(image.max()-image.min())
    kmeans = KMeans(n_clusters=kmeans_n_clusters, algorithm="lloyd").fit(image_normalised)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    segmented_image = centers[labels]
    segmented_normalized = (segmented_image - segmented_image.min())/(segmented_image.max()-segmented_image.min())
    rim_binarized = segmented_normalized>=segmented_threshold
    points = np.argwhere(rim_binarized)
    return points, rim_binarized

def _ransac_circle(points:np.ndarray, n_samples:int=3, n_iterations:int=1000, threshold:float=0.1) -> tuple[CircleModel, np.ndarray]:
    from skimage.measure import CircleModel, ransac
    model, inliers = ransac(points, CircleModel, min_samples=n_samples, max_trials=n_iterations, residual_threshold=threshold)
    return model, inliers   

def _estimate_center(points:np.ndarray):
    # ell = EllipseModel()
    # succ = ell.estimate(points)
    # print(succ)
    # # Draw the ellipse on the original image
    # ye, xe, a, b = (int(round(x)) for x in ell.params[:-1])
    # ey, ex = ellipse_perimeter(ye,xe,a,b,ell.params[-1])

    # circ = CircleModel()
    # success = circ.estimate(points)
    # if not success:
    #     return None

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

def _radial_profile(image:np.ndarray, center:tuple, pixelscale:float, max_radius:str="inner radius"):
    dipimg = dip.Image(image)
    rad = dip.RadialMean(dipimg, binSize=1, center=center, maxRadius=max_radius)
    prof = np.asarray(rad)
    x_ax = np.arange(len(prof))
    # pixelscale = 0.040639999999999996e-6
    x_ax = x_ax * pixelscale
    return np.vstack((x_ax,prof)) 

def extract_ridge_from_image(image_file: Path, kmeans_n_clusters=50, segmented_threshold=0.5):
    if isinstance(image_file, str):
        image_file = Path(image_file)
    with tifffile.TiffFile(image_file) as tif:
        image_grey = tif.asarray()
        ome_metadata = tif.ome_metadata
        metadata = tif.pages[0].tags
        # The TIFF types of the XResolution and YResolution tags are RATIONAL (5)
        # which is defined in the TIFF specification as two longs, the first of which is the numerator and the second the denominator.
        xres = Fraction(*metadata["XResolution"].value)
        yres = Fraction(*metadata["YResolution"].value)
        pixelscale = (1/xres + 1/yres) / 2
            
    points, rim = _rim_from_image(image_grey, kmeans_n_clusters=kmeans_n_clusters, segmented_threshold=segmented_threshold)

    yc, xc, r, r2 = _estimate_center(points)
    print(f"{image_file.stem} Center: ({yc},{xc}) Radius: {r} R2: {r2}")

    radial_profile = _radial_profile(image_grey, (yc,xc), pixelscale, max_radius="inner radius")
    return radial_profile, image_grey, (yc,xc,r), rim


