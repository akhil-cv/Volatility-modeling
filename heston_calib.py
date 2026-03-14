#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# heston_calib.py
import numpy as np
from math import log, exp, sqrt, pi
from dataclasses import dataclass
from typing import Sequence, Tuple
from scipy import integrate, optimize, stats
from scipy.optimize import brentq

# -----------------------
# Black-Scholes helpers
# -----------------------
def bs_price_call(S, K, T, r, q, sigma):
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)

def implied_vol_from_price(mkt_price, S, K, T, r, q, vol_bounds=(1e-6, 5.0)):
    # safe bracketed root-finding for implied vol (Brent)
    if mkt_price <= max(0.0, S * np.exp(-q*T) - K * np.exp(-r*T)) + 1e-12:
        return 0.0
    def f(sigma):
        return bs_price_call(S, K, T, r, q, sigma) - mkt_price
    a, b = vol_bounds
    try:
        return brentq(f, a, b, maxiter=100, xtol=1e-8)
    except Exception:
        # fallback: return NaN on failure
        return np.nan

# -----------------------
# Heston characteristic function & price integration
# -----------------------
@dataclass
class HestonParams:
    V0: float   # initial variance
    kappa: float
    theta: float
    eta: float
    rho: float

def _heston_char(u: complex, params: HestonParams, S: float, r: float, q: float, T: float, phi_type: int):
    """
    Returns characteristic function value phi(u) for Heston (following standard formulation).
    phi_type in {1,2} picks the integration formula variant for P1 and P2.
    """
    V0, kappa, theta, eta, rho = params.V0, params.kappa, params.theta, params.eta, params.rho

    # complex i
    i = 1j

    # phi_type affects parameters a, b, u_shift as in Heston formulation
    if phi_type == 1:
        u_shift = u
        alpha = 0.5
        beta = kappa - rho * eta
    else:
        u_shift = u
        alpha = -0.5
        beta = kappa

    # Standard Heston definitions
    a = kappa * theta
    b = kappa - rho * eta * i * u_shift
    # discriminant
    sigma = eta
    d = np.sqrt(b * b + (u_shift * i + u_shift * u_shift) * sigma * sigma)
    g = (b - d) / (b + d)

    # Avoid branch-cut issues: use exp in stable form
    exp_dt = np.exp(-d * T)

    # C and D following commonly used form
    C = (r - q) * i * u_shift * T + a / (sigma * sigma) * ((b - d) * T - 2.0 * np.log((1.0 - g * exp_dt) / (1.0 - g)))
    D = (b - d) / (sigma * sigma) * (1.0 - exp_dt) / (1.0 - g * exp_dt)

    # characteristic function
    phi = np.exp(C + D * V0 + i * u_shift * np.log(S * np.exp(-q * T)))
    return phi

def _P_integral(S, K, T, r, q, params: HestonParams, phi_type: int, integration_limit=200.0):
    """
    Compute P_j by numerical integration:
    P_j = 1/2 + 1/pi * \int_0^\infty Re( e^{-i u ln K} * phi(u) / (i u) ) du
    """
    lnK = np.log(K)

    def integrand(u):
        if u == 0.0:
            return 0.0
        phi = _heston_char(u - 1j * 0.0, params, S, r, q, T, phi_type)
        numerator = np.exp(-1j * u * lnK) * phi
        denom = 1j * u
        val = numerator / denom
        return np.real(val)

    # integrate from 0 to +inf; use an upper limit large enough
    val, err = integrate.quad(integrand, 0.0, integration_limit, limit=200, epsabs=1e-6, epsrel=1e-5)
    P = 0.5 + val / pi
    return P

def heston_call_price(S, K, T, r, q, params: HestonParams, integration_limit=200.0):
    """
    Price a European call option under Heston using P1/P2 integration.
    """
    if T <= 0:
        return max(S - K, 0.0)
    # compute P1 and P2
    P1 = _P_integral(S, K, T, r, q, params, phi_type=1, integration_limit=integration_limit)
    P2 = _P_integral(S, K, T, r, q, params, phi_type=2, integration_limit=integration_limit)
    # discounted forward
    F = S * np.exp(-q * T)
    price = np.exp(-r * T) * (F * P1 - K * P2 * np.exp(r * T))  # rearranged to standard
    # simpler: price = S*exp(-qT)*P1 - K*exp(-rT)*P2
    price = S * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
    return np.real_if_close(price)

# -----------------------
# Calibration routine
# -----------------------
def calibrate_heston_implied_vols(market_data: Sequence[dict],
                                   spot: float,
                                   r: float,
                                   q: float = 0.0,
                                   initial_guess=None,
                                   bounds=None,
                                   integration_limit=200.0,
                                   verbose=False):
    """
    Calibrate Heston parameters to market implied volatilities.

    market_data: sequence of dicts, each dict contains:
        {
          "K": strike,
          "T": maturity_in_years,
          "iv": market_implied_vol
        }
    spot: current spot S
    r: risk-free rate (annual)
    q: dividend yield (annual)
    initial_guess: list-like [V0, kappa, theta, eta, rho]
    bounds: tuple (lower_bounds, upper_bounds) each list-like
    integration_limit: upper limit for integrals (increase for heavier tails)
    """

    # Build arrays
    Ks = np.array([d["K"] for d in market_data], dtype=float)
    Ts = np.array([d["T"] for d in market_data], dtype=float)
    ivs = np.array([d["iv"] for d in market_data], dtype=float)

    # convert iv -> market prices
    market_prices = np.array([bs_price_call(spot, K, T, r, q, iv) for K, T, iv in zip(Ks, Ts, ivs)])

    # default initial guess
    if initial_guess is None:
        median_iv = np.nanmedian(ivs[np.isfinite(ivs)])
        v0_guess = (median_iv if np.isfinite(median_iv) else 0.2) ** 2
        initial_guess = [v0_guess, 1.0, v0_guess, 0.5, -0.5]

    # default bounds
    if bounds is None:
        lb = [1e-6, 1e-6, 1e-6, 1e-6, -0.999]
        ub = [5.0, 20.0, 5.0, 5.0, 0.999]
    else:
        lb, ub = bounds

    # residual function for least_squares
    def residuals(x):
        V0, kappa, theta, eta, rho = x
        params = HestonParams(V0=max(1e-12, V0),
                              kappa=max(1e-12, kappa),
                              theta=max(1e-12, theta),
                              eta=max(1e-12, eta),
                              rho=np.clip(rho, -0.999, 0.999))
        model_prices = []
        for K, T in zip(Ks, Ts):
            try:
                p = heston_call_price(spot, K, T, r, q, params, integration_limit=integration_limit)
            except Exception:
                p = np.nan
            model_prices.append(p)
        model_prices = np.array(model_prices)

        # Convert model prices back to implied vols to compare in vol space (preferred)
        model_ivs = np.array([implied_vol_from_price(p, spot, K, T, r, q) if np.isfinite(p) else np.nan
                              for p, K, T in zip(model_prices, Ks, Ts)])
        # residual: difference in vols (ignore NaNs)
        resid = model_ivs - ivs
        # replace NaNs by large penalty (so optimizer avoids)
        resid = np.where(np.isfinite(resid), resid, 10.0)
        return resid

    # use least_squares (fast local)
    result = optimize.least_squares(residuals, x0=initial_guess, bounds=(lb, ub), xtol=1e-8, ftol=1e-8, verbose=2 if verbose else 0, max_nfev=200)

    V0_cal, kappa_cal, theta_cal, eta_cal, rho_cal = result.x
    params_cal = HestonParams(V0=max(1e-12, V0_cal),
                              kappa=max(1e-12, kappa_cal),
                              theta=max(1e-12, theta_cal),
                              eta=max(1e-12, eta_cal),
                              rho=np.clip(rho_cal, -0.999, 0.999))

    # Compute final model implied vols & RMSE
    model_prices_final = np.array([heston_call_price(spot, K, T, r, q, params_cal, integration_limit=integration_limit) for K, T in zip(Ks, Ts)])
    model_ivs_final = np.array([implied_vol_from_price(p, spot, K, T, r, q) for p, K, T in zip(model_prices_final, Ks, Ts)])
    mask = np.isfinite(model_ivs_final)
    rmse = np.sqrt(np.nanmean((model_ivs_final[mask] - ivs[mask]) ** 2)) if np.any(mask) else np.nan

    return {
        "params": params_cal,
        "model_ivs": model_ivs_final,
        "market_ivs": ivs,
        "strikes": Ks,
        "maturities": Ts,
        "rmse": rmse,
        "opt_result": result
    }

# -----------------------
# Example usage:
# -----------------------
if __name__ == "__main__":
    # Example market data (synthetic). Replace with your real implied vol surface.
    spot = 100.0
    r = 0.01
    q = 0.0
    market_data = [
        {"K": 80.0, "T": 0.08, "iv": 0.35},
        {"K": 90.0, "T": 0.08, "iv": 0.30},
        {"K": 100.0, "T": 0.08, "iv": 0.25},
        {"K": 110.0, "T": 0.08, "iv": 0.28},
        {"K": 120.0, "T": 0.08, "iv": 0.33},
        {"K": 80.0, "T": 0.5, "iv": 0.30},
        {"K": 100.0, "T": 0.5, "iv": 0.22},
        {"K": 120.0, "T": 0.5, "iv": 0.27},
    ]

    out = calibrate_heston_implied_vols(market_data, spot, r, q=q, verbose=True, integration_limit=150.0)
    print("Calibrated params:", out["params"])
    print("RMSE (vol):", out["rmse"])
    for K, T, m_iv, mod_iv in zip(out["strikes"], out["maturities"], out["market_ivs"], out["model_ivs"]):
        print(f"K={K:6.1f}, T={T:.3f}  market_iv={m_iv:.4f} model_iv={mod_iv:.4f}")

