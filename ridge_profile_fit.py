import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
from data_types import RidgeFit, RidgeFitMethod
from numpy.polynomial.polynomial import Polynomial

from ridge_models import style_exact, style_ld_ts, shanahan, limat_symmetric

def calc_peak_angle(x, y, peak_idx):
    try:
        line_up =  Polynomial.fit(x[peak_idx-4:peak_idx+1], y[peak_idx-4:peak_idx+1], 1).convert().coef[1]
        line_down = Polynomial.fit(x[peak_idx:peak_idx+5], y[peak_idx:peak_idx+5], 1).convert().coef[1]

        return np.rad2deg(np.arctan2(line_down,1) - np.arctan2(line_up,1))
    except:
        return np.nan


def fit_profile(file, gamma, upsilon, gamma_s, theta, E_lookup, h, models=["all"]) -> dict[str, RidgeFit]:

    # profile = pd.read_csv(file, sep=";", skiprows=3, names=["x","y","idk"], index_col=False)
    profile = pd.read_csv(file, sep=";", index_col=False)

    Fe, G, vol, th, *_ = Path(file).stem.split("_")

    if "all" in models:
        models = ["style", "style_ld", "shanahan", "limat"]

    E = E_lookup[f"{Fe}_{G}"]   
    h = h[th]

    defl = np.asarray(profile["y"], dtype=np.float64)[50:]
    x_ax = np.asarray(profile["x"], dtype=np.float64)[50:]
    R = x_ax[np.argmax(defl[10:])+10]

    peak = np.argmax(defl)
    x_ax += R - x_ax[peak]

    x_peak_ps = x_ax[peak-1:peak+2]

    def fit_wrap(func):
        def f(x, p1, p2, p3):
            return func(x, gamma, R, p1, p2, p3)
        return f
    
    fits = dict()
    
    print(f"{Fe}_{G}_{vol} - {R*1000:.1f}")

# calculate the angle below the peak theta_s
    # peak_angle = np.rad2deg(np.arctan2(defl[peak-1]-defl[peak], x_ax[peak-1]-x_ax[peak])- 
    #                         np.arctan2(defl[peak+1]-defl[peak], x_ax[peak+1]-x_ax[peak]))
    peak_angle = calc_peak_angle(x_ax, defl, peak)

    fits["actual"] = RidgeFit(x_ax, defl, vol, R, G, Fe, gamma, RidgeFitMethod.NONE, None, [-1, E, -1], None, peak_angle=peak_angle)

    curve_fit_lsq_args = {"nan_policy":"omit","max_nfev": 400*(1+len(x_ax)), "loss":"soft_l1", "full_output":True, "gtol":1e-9, "xtol":1e-9, "ftol":1e-9, "method":"trf"}

    if "shanahan" in models:
        popt, pcov, infodict, mesg, ier = curve_fit(fit_wrap(shanahan), x_ax, defl, p0=(theta, E, 1), bounds=([0,-np.inf,0],[180,np.inf,np.inf]), **curve_fit_lsq_args)
        # y_vals = shanahan(x_ax[peak-1:peak+2], gamma, R, *popt)
        # peak_angle = np.rad2deg(np.arctan2(y_vals[0]-y_vals[1], x_peak_ps[0]-x_peak_ps[1])- 
        #                         np.arctan2(y_vals[2]-y_vals[1], x_peak_ps[0]-x_peak_ps[1]))
        peak_angle = calc_peak_angle(x_ax, shanahan(x_ax, gamma, R, *popt), peak)
        fits["shanahan"] = RidgeFit(x_ax, defl, vol, R, G, Fe, gamma, RidgeFitMethod.SHANAHAN, shanahan, popt, pcov, peak_angle=peak_angle)
        print(f"Shanahan:\tR2 {fits["shanahan"].r2:.3f}")
        print(mesg)

    if "limat" in models:
        popt, pcov, infodict, mesg, ier = curve_fit(fit_wrap(limat_symmetric), x_ax, defl, p0=(gamma_s, E, theta), bounds=([0,0,0],[np.inf,np.inf,180]), **curve_fit_lsq_args)
        # y_vals = limat_symmetric(x_ax[peak-1:peak+2], gamma, R, *popt)
        # peak_angle = np.rad2deg(np.arctan2(y_vals[0]-y_vals[1], x_peak_ps[0]-x_peak_ps[1])- 
        #                         np.arctan2(y_vals[2]-y_vals[1], x_peak_ps[0]-x_peak_ps[1]))
        peak_angle = calc_peak_angle(x_ax, limat_symmetric(x_ax, gamma, R, *popt), peak)
        fits["limat"] = RidgeFit(x_ax, defl, vol, R, G, Fe, gamma, RidgeFitMethod.LIMAT, limat_symmetric, popt, pcov, peak_angle=peak_angle)
        print(f"Limat:\t\tR2 {fits["limat"].r2:.3f}")
        print(mesg)

    if "style_ld" in models:
        popt, pcov, infodict, mesg, ier = curve_fit(fit_wrap(style_ld_ts), x_ax, defl, p0=(upsilon, E, h), bounds=([0,0,0],[np.inf,np.inf,np.inf]), **curve_fit_lsq_args)
        # y_vals = style_ld_ts(x_ax[peak-1:peak+2], gamma, R, *popt)
        # peak_angle = np.rad2deg(np.arctan2(y_vals[0]-y_vals[1], x_peak_ps[0]-x_peak_ps[1])- 
        #                         np.arctan2(y_vals[2]-y_vals[1], x_peak_ps[0]-x_peak_ps[1]))
        peak_angle = calc_peak_angle(x_ax, style_ld_ts(x_ax, gamma, R, *popt), peak)
        fits["style_ld"] = RidgeFit(x_ax, defl, vol, R, G, Fe, gamma, RidgeFitMethod.STYLE_LD, style_ld_ts, popt, pcov, peak_angle=peak_angle)
        print(f"Style R>>h:\tR2 {fits["style_ld"].r2:.3f}")
        print(mesg)

    if "style" in models:
        popt, pcov, infodict, mesg, ier = curve_fit(fit_wrap(style_exact), x_ax, defl, p0=(upsilon, E, h), bounds=([0,0,0],[np.inf,np.inf,np.inf]), **curve_fit_lsq_args)
        # y_vals = style_exact(x_ax[peak-1:peak+2], gamma, R, *popt)
        # peak_angle = np.rad2deg(np.arctan2(y_vals[0]-y_vals[1], x_peak_ps[0]-x_peak_ps[1])- 
        #                         np.arctan2(y_vals[2]-y_vals[1], x_peak_ps[0]-x_peak_ps[1]))
        peak_angle = calc_peak_angle(x_ax, style_exact(x_ax, gamma, R, *popt), peak)
        fits["style"] = RidgeFit(x_ax, defl, vol, R, G, Fe, gamma, RidgeFitMethod.STYLE_LD, style_exact, popt, pcov, peak_angle=peak_angle)
        print(f"Style:\t\tR2 {fits["style"].r2:.3f}")
        print(mesg)
    print()

    return fits

def fit_profile_style(file, gamma, upsilon,  E_lookup, h, fix_upsilon = False, fix_h = False, fix_E=False, **kwargs) -> dict[str, RidgeFit]:

    # profile = pd.read_csv(file, sep=";", skiprows=3, names=["x","y","idk"], index_col=False)
    profile = pd.read_csv(file, sep=";", index_col=False)

    Fe, G, vol, th, *_ = Path(file).stem.split("_")

    E = E_lookup[f"{Fe}_{G}"]   
    h = h[th]

    defl = np.asarray(profile["y"], dtype=np.float64)[50:]
    x_ax = np.asarray(profile["x"], dtype=np.float64)[50:]
    R = x_ax[np.argmax(defl[10:])+10]

    peak = np.argmax(defl)
    # x_ax += R - x_ax[peak]

    # x_peak_ps = x_ax[peak-1:peak+2]

    def fit_wrap(func):
        def f(x, p1, p2, p3):
            return func(x, gamma, R, p1, p2, p3)
        return f
    
    fits = dict()
    
    print(f"{Fe}_{G}_{vol} - {R*1000:.1f}")

    # calculate the angle below the peak theta_s

    peak_angle = calc_peak_angle(x_ax, defl, peak)

    fits["actual"] = RidgeFit(x_ax, defl, vol, R, G, Fe, gamma, RidgeFitMethod.NONE, None, [-1, E, -1], None, peak_angle=peak_angle)

    curve_fit_lsq_args = {"nan_policy":"omit","max_nfev": 400*(1+len(x_ax)), "loss":"soft_l1", "full_output":True, "gtol":1e-9, "xtol":1e-9, "ftol":1e-9, "method":"trf",
                          "bounds":([0,0,0],[np.inf,np.inf,np.inf]), "p0":(upsilon, E, h)}

    if fix_upsilon:
        curve_fit_lsq_args["bounds"][0][0] = upsilon-np.finfo(float).eps
        curve_fit_lsq_args["bounds"][1][0] = upsilon
    if fix_h:
        curve_fit_lsq_args["bounds"][0][2] = h-np.finfo(float).eps
        curve_fit_lsq_args["bounds"][1][2] = h
    if fix_E:
        curve_fit_lsq_args["bounds"][0][1] = E-1
        curve_fit_lsq_args["bounds"][1][1] = E
    
    popt, pcov, infodict, mesg, ier = curve_fit(fit_wrap(style_exact), x_ax, defl, **curve_fit_lsq_args)
    peak_angle = calc_peak_angle(x_ax, style_exact(x_ax, gamma, R, *popt), peak)
    fits["style"] = RidgeFit(x_ax, defl, vol, R, G, Fe, gamma, RidgeFitMethod.STYLE_LD, style_exact, popt, pcov, peak_angle=peak_angle)
    print(f"Style:\t\tR2 {fits["style"].r2:.3f}")
    print(mesg)
    print()

    return fits