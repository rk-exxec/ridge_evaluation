import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
from data_types import RidgeFit, RidgeFitMethod


from ridge_models import style_exact, style_ld_ts, shanahan, limat_symmetric


def fit_profile(file, gamma, upsilon, gamma_s, theta, E_lookup, h):

    # profile = pd.read_csv(file, sep=";", skiprows=3, names=["x","y","idk"], index_col=False)
    profile = pd.read_csv(file, sep=";", index_col=False)

    Fe, G, vol, _ = Path(file).stem.split("_")

    E = E_lookup[f"{Fe}_{G}"]

    defl = np.asarray(profile["y"], dtype=np.float64)[50:]
    x_ax = np.asarray(profile["x"], dtype=np.float64)[50:]
    R = x_ax[np.argmax(defl[100:])+100]

    peak = np.argmax(defl)
    x_ax += R - x_ax[peak]

    def fit_wrap(func):
        def f(x, p1, p2, p3):
            return func(x, gamma, R, p1, p2, p3)
        return f
    
    fits = dict()
    
    print(f"{Fe}_{G}_{vol} - {R*1000:.1f}")

    popt, pcov = curve_fit(fit_wrap(shanahan), x_ax, defl, p0=(theta, 1, E), bounds=([0,-np.inf,0],[180,np.inf,np.inf]))
    fits["shanahan"] = RidgeFit(x_ax, defl, vol, R, G, Fe, gamma, RidgeFitMethod.SHANAHAN, shanahan, popt, pcov)
    print(f"Shanahan:\tR2 {fits["shanahan"].r2:.3f}")

    popt, pcov = curve_fit(fit_wrap(limat_symmetric), x_ax, defl, p0=(gamma_s, theta, E), bounds=([0,0,0],[np.inf,180,np.inf]))
    fits["limat"] = RidgeFit(x_ax, defl, vol, R, G, Fe, gamma, RidgeFitMethod.LIMAT, limat_symmetric, popt, pcov)
    print(f"Limat:\t\tR2 {fits["limat"].r2:.3f}")

    popt, pcov = curve_fit(fit_wrap(style_ld_ts), x_ax, defl, p0=(upsilon, E, h), bounds=([0,0,0],[np.inf,np.inf,np.inf]))
    fits["style_ld"] = RidgeFit(x_ax, defl, vol, R, G, Fe, gamma, RidgeFitMethod.STYLE_LD, style_ld_ts, popt, pcov)
    print(f"Style R>>h:\tR2 {fits["style_ld"].r2:.3f}")

    popt, pcov = curve_fit(fit_wrap(style_exact), x_ax, defl, p0=(upsilon, E, h), bounds=([0,0,0],[np.inf,np.inf,np.inf]))
    fits["style"] = RidgeFit(x_ax, defl, vol, R, G, Fe, gamma, RidgeFitMethod.STYLE_LD, style_exact, popt, pcov)
    print(f"Style:\t\tR2 {fits["style"].r2:.3f}")
    print()

    return fits