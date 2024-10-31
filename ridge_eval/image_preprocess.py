import numpy as np

def ra(arr: np.ndarray):
    mean_z = np.nanmean(arr, dtype=np.float64)
    total_count = np.count_nonzero(~np.isnan(arr))
    diffs = np.abs(arr - mean_z) #np.subtract(arr, mean_z, where=~np.isnan(arr))
    
    return np.nansum(diffs) / total_count

def rq(arr: np.ndarray):
    mean_z = np.nanmean(arr, dtype=np.float64)
    total_count = np.count_nonzero(~np.isnan(arr))
    diffs = np.abs(arr - mean_z) #np.subtract(arr, mean_z, where=~np.isnan(arr))
    return np.sqrt(np.nansum(diffs**2) / total_count)

def mean_plane_leveling(arr: np.ndarray, resolution) -> np.ndarray:
    ax.plot_surface(X1,X2,Y, rstride = 1, cstride = 1, cmap = jet, linewidth = 0)
    shape = arr.shape
    m, n = shape
    X1,X2 = np.mgrid[:m, :n]
    Y = arr
    X = np.hstack(   ( np.reshape(X1, (m*n, 1)) , np.reshape(X2, (m*n, 1)) ) )
    X = np.hstack(   ( np.ones((m*n, 1)) , X ))
    YY = np.reshape(Y, (m*n, 1))

    theta = np.dot(np.dot( np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), YY)

    plane = np.reshape(np.dot(X, theta), (m, n))

    ax = fig.add_subplot(3,1,2, projection='3d')
    ax.plot_surface(X1,X2,plane)
    ax.plot_surface(X1,X2,Y, rstride = 1, cstride = 1, cmap = jet, linewidth = 0)