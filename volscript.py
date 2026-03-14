#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datetime
import os
from typing import List
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tseries.offsets import BDay

# from scriptaman_counterparty.risk.equity.simulations.local_volatility.model_calibration.calibrate.local_vol_model_pa import (
#     get_spot_varswap_hts_from_tss,
#     get_instantaneous_variance
# )

np.random.seed(1000)

def smoothed_maximum(x, c):
    return 0.5 * (x + np.sqrt(np.power(x, 2) + np.power(c, 2)))

def plot_simulation_paths(df, folder, file_name):
    """
    Plot simulation paths using matplotlib
    @param df: DataFrame containing simulation paths
    @param folder: Folder to save the plot
    @param file_name: Name of the output file
    @return: None
    """
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111)
    ax.plot(df.transpose())
    plt.xlabel("time step")
    plt.ylabel("spot")
    plt.title("Spot trajectories")
    plt.savefig(os.path.join(folder, file_name))
    plt.close()


def expo_scheme(prev_spot, drift, sigma, dt, rand):
    """
    Doob-Meyer Exponential simulation scheme
    @param prev_spot: Previous spot value
    @param drift: Drift term
    @param sigma: Volatility
    @param dt: Time increment
    @param rand: Random shock
    @return: Simulated spot value
    """
    return prev_spot * np.exp((drift - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rand)

def euler_scheme(prev_spot, drift, sigma, dt, rand):
    """
    Euler simulation scheme
    @param prev_spot: Previous spot value
    @param drift: Drift term
    @param sigma: Volatility
    @param dt: Time increment
    @param rand: Random shock
    @return: Simulated spot value
    """
    return prev_spot * (1 + drift * dt + sigma * np.sqrt(dt) * rand)

def compute_equiv_local_vol_parameters(v0, kappa, theta, eta, rho, t):
    """
    Calibrate equivalent local volatility parameters (alpha and beta) from Heston parameters
    @param v0: Initial variance
    @param kappa: Mean reversion speed
    @param theta: Long-term variance
    @param eta: Volatility of volatility
    @param rho: Correlation between asset and volatility
    @param t: Time horizon
    @return: alpha, beta, gamma
    """
    if t == 0:
        return None, None
    else:
        lambda_ = kappa - 0.5 * rho * eta
        alpha = (v0 - theta * kappa / lambda_) * np.exp(-lambda_ * t) + theta * kappa / lambda_
        
        beta_num = ((v0 - theta) * (np.exp(-lambda_ * t) - np.exp(-kappa*t)) / (kappa - lambda_) + theta * (
            1- np.exp(-lambda_ * t)) / lambda_)
        
        beta_den = (v0 - theta) * (1- np.exp(-kappa*t)) / kappa + theta * t
        beta = rho * eta * beta_num / beta_den
        gamma = 0
        return alpha, beta, gamma

    
def compute_equiv_local_vol_parameters_v2(V0, kappa, theta, eta, rho, t):
    """
    Calibrate equivalent local volatility parameters (alpha, beta, gamma) from Heston parameters
    @param V0: Initial variance
    @param kappa: Mean reversion speed
    @param theta: Long-term variance
    @param eta: Volatility of volatility
    @param rho: Correlation between asset and volatility
    @param t: Time horizon
    @return: alpha, beta, gamma
    """
    if t == 0:
        return None, None
    else:
        lambda_ = kappa - 0.5 * rho * eta

        lambda_ = kappa - 0.5 * rho * eta
        alpha = (V0 - theta * kappa / lambda_) * np.exp(-lambda_ * t) + theta * kappa / lambda_
        
        beta_num = ((V0 - theta) * (np.exp(-lambda_ * t) - np.exp(-kappa*t)) / (kappa - lambda_) + theta * (
            1- np.exp(-lambda_ * t)) / lambda_)
        
        beta_den = (V0 - theta) * (1- np.exp(-kappa*t)) / kappa + theta * t
        second_order_correction = np.sqrt(1- rho**2)* eta * (1-np.exp(-lambda_ *t)) / lambda_
        beta = rho * eta * beta_num / beta_den + second_order_correction
        
        gamma = np.sqrt((1 - rho **2) * eta * g(V0, kappa, theta, eta, rho, t) * (1 - np.exp(-lambda_ * t)) / lambda_)

        return alpha, beta, gamma

def compute_equiv_local_vol_parameters_V3(V0, kappa, theta, eta, rho, t):
    """
    Calculate equivalent local volatility parameters (alpha, beta, gamma) from Heston parameters

    @param v0: Initial variance
    @param kappa: Mean reversion speed
    @param theta: Long-term variance
    @param eta: Volatility of volatility
    @param rho: Correlation between asset and volatility
    @param t: Time horizon
    @return: alpha, beta, gamma
    """
    if t == 0:
        return None
    else:
        lambda_ = kappa - 0.5 * rho * eta

        alpha = (V0 - theta * kappa / lambda_) * np.exp(-lambda_ * t) + theta * kappa / lambda_

        beta_num = ((V0 - theta) * (np.exp(-lambda_ * t) - np.exp(-kappa*t)) / (kappa - lambda_) + theta * (
            1- np.exp(-lambda_ * t)) / lambda_)
        
        beta_den = (V0 - theta) * (1- np.exp(-kappa*t)) / kappa + theta * t
        beta = rho * eta * beta_num / beta_den 
        
        gamma = np.sqrt((1 - rho **2) * eta * 0.5/t)

        return alpha, beta, gamma

def g(v0, kappa, theta, eta, rho, t):
    numerator = 0.5 * (v0 - theta) * (1 - np.exp(-kappa * t)) / kappa + theta * t
    return numerator / expectation_x_square(v0, kappa, theta, eta, rho, t)

def expectation_x_square(V0, kappa, theta, eta, rho, t):
    c_0 = pow(V0 - theta, 2) / (4 * pow(kappa, 2)) - (V0 - theta) * (2 * eta * rho - pow(eta, 2) / kappa) / (2 * pow(kappa, 2)) + \
        (eta * rho * theta) / pow(kappa, 2) - (V0*pow(eta,2))/(4*pow(kappa,3)) - \
        (theta * pow(eta,2)) / (8*pow(kappa, 3)) +(V0 -theta) / kappa

    c_1 = theta * (2 * eta * rho - pow(eta, 2) / (2 * kappa) - (V0 - theta) - 2* kappa) / (2 * kappa)
    
    c_2 = pow(theta,2) / 4.0
    
    c_3 = -pow(V0 -theta,2) / (2*pow(kappa,2)) + eta * (V0 - theta)* (2*rho - eta / kappa) / (2* pow(kappa,2)) + \
        pow(eta,2) * V0 / (2*pow(kappa,3)) - eta * rho *theta / pow(kappa,2) - (V0 - theta) / kappa

    c_4 = (pow(V0 - theta, 2) -pow(eta, 2) * (2.0 * V0 - theta) / (2* kappa)) / (4 * pow(kappa, 2))
    
    c_5 = (V0 - theta) * (eta*rho - theta/2.0 - pow(eta,2)/ (2*kappa))/kappa

    denominator = (
        c_0 + c_1 + c_2 * pow(t, 2) +
        c_3 * np.exp(-kappa * t) +
        c_4 * np.exp(-2 * kappa * t) +
        c_5 * t * np.exp(-kappa * t)
    )

    return denominator


def compute_implied_vol_parameters(v0, kappa, theta, eta, rho, maturity):
    """
    Calibrate implied volatility parameters (A and B) from Heston parameters.

    Parameters:
    - v0: initial variance
    - kappa: rate of mean reversion
    - theta: long-term variance
    - eta: volatility of volatility
    - rho: correlation between asset and variance
    - maturity: time to maturity

    Returns:
    - A, B: implied volatility parameters
    """
    lambda_ = kappa - 0.5 * rho * eta
    u = maturity
    increment = (1.0 - np.exp(-lambda_ * u)) / (lambda_ * u)

    A = (v0 - theta * kappa / lambda_) * increment + theta * kappa / lambda_
    B = (rho * eta / (lambda_ * u)) * (1.0 - increment)

    return A, B

def get_realised_volatility(df):
    """
    Get realised volatility from simulated spot trajectories (for daily time-steps only)
    @param df:
    @return:
    """
    # Compute log-returns
    returns_df = (np.log(df.astype('float')) - np.log(df.astype('float')).shift(1, axis=1)).loc[:,
                 df.columns != df.columns.values[0]]

    # Get realised volatility of each trajectory
    annualization_factor = len(returns_df.columns.values) / 260.0
    variances_df = (returns_df.apply(lambda x: pow(x, 2)).sum(axis=1)) / annualization_factor
    std_df = np.sqrt(variances_df)

    return std_df


def get_accumulated_realised_volatility_stats(df):
    """
    Get realised volatility from simulated spot trajectories (for daily time-steps only)
    @param df:
    @return:
    """
    # Compute log-returns
    returns_squared_df = ((np.log(df.astype('float')) / np.log(df.astype('float')).shift(1, axis=1)).loc[:,
                      df.columns != df.columns.values[0]]).apply(lambda x: pow(x, 2))

    # Get realised volatility of each trajectory
    accumulated_variance = pd.Series(index=returns_squared_df.index, data=[0]*returns_squared_df.shape[0])
    variances_df = pd.DataFrame(index=returns_squared_df.index, columns=returns_squared_df.columns.values)

    nb_days = 1.0
    for column_name, column_data in returns_squared_df.iteritems():
        annualization_factor = nb_days /260.0
        accumulated_variance = accumulated_variance.add(column_data)
        annualised_accumulated_variance = (accumulated_variance / annualization_factor)
        variances_df[:, column_name] = annualised_accumulated_variance
        nb_days += 1.0

    std_df = np.sqrt(variances_df)

    # Compute statistics on accumulated variance distribution
    quantile_1_df = std_df.copy().quantile(0.01, axis=0)
    quantile_99_df = std_df.copy().quantile(0.99, axis=0)
    average_df = std_df.copy().mean(axis=0)
    std_std_df = std_df.copy().std(axis =0)
    results = pd.concat([quantile_1_df, quantile_99_df, average_df, std_std_df], axis=1)
    return results

def get_stats_spot_distribution(df):
    """
    Get stats about spot distribution over all trajectories (for each daily time-steps)
    @param df:
    @return:
    Stats about spot distribution
    """
    quantile_1_df = df.copy().quantile(0.01, axis=0)
    # quantile_5_df = df.copy().quantile(0.05, axis=0)
    quantile_10_df = df.copy().quantile(0.1, axis=0)
    quantile_25_df = df.copy().quantile(0.25, axis=0)
    quantile_50_df = df.copy().quantile(0.5, axis=0)
    quantile_75_df = df.copy().quantile(0.75, axis=0)
    quantile_90_df = df.copy().quantile(0.9, axis=0)
    quantile_95_df = df.copy().quantile(0.95, axis=0)
    quantile_99_df = df.copy().quantile(0.99, axis=0)
    average_df = df.copy().mean(axis=0)

    results = pd.concat([quantile_1_df, quantile_10_df, quantile_25_df, quantile_50_df, quantile_75_df, quantile_90_df, quantile_99_df], axis=1)

    # Histogram related to distribution at maturity
    spot_distrib_maturity = df[df.columns.values[-1]]
    return results, spot_distrib_maturity

class SpotDiffusion(ABC):
    """
    Abstract base class to simulate equity spot risk-factor
    """

    spot: float
    drift: float
    time_points: List[float]
    number_points: float

    @abstractmethod
    def get_simulation_volatility(self, time_index, spot_i):
        pass

    @abstractmethod
    def get_simulation_drift(self, time_index):
        pass

    def generate_paths(self, scheme="euler"):
        """
        Generate spot simulations matrix
        @param scheme:
        @return:
        """
        spot_simulations = np.zeros((len(self.time_points), self.number_paths), np.float64)
        spot_simulations[0] = self.spot

        for i in range(1, len(self.time_points)):
            
            rand = np.random.standard_normal(self.number_paths)
            vol = self.get_simulation_volatility(i - 1, spot_simulations[i - 1])
            drift = self.get_simulation_drift(i -1)
            dt = (self.time_points[i] - self.time_points[i - 1])

            # Use euler simulation scheme
            if scheme == "euler":
                spot_simulations[i] = euler_scheme(spot_simulations[i - 1], drift, vol, dt, rand)
            # Else use doob-danbe exponential scheme
            else:
                spot_simulations[i] = expo_scheme(spot_simulations[i - 1], drift, vol, dt, rand)

            self.v_simulations[i-1] = vol * vol

        self.v_simulations[len(self.time_points) - 1] = np.power(
            self.get_simulation_volatility(len(self.time_points) - 1 ,spot_simulations[len(self.time_points)-2]), 2.0)

        result = pd.DataFrame(spot_simulations.transpose())
        result.columns = self.time_points
        return result

    def get_v_inst_simulations(self):
        return pd.DataFrame(self.v_simulations.transpose())

class ConstantVolatilityModel(SpotDiffusion):
    # constant volatility model (Black-Scholes like)

    volatility: float

    def __init__(self, spot, volatility, drift, time_points, number_paths):
        self.spot = spot
        self.volatility = volatility
        self.drift = drift
        self.time_points = time_points
        self.number_paths = number_paths
        self.v_simulations = np.zeros((len(self.time_points), self.number_paths), np.float64)

    def get_simulation_volatility(self, time_index, spot_i):
        '''
        # Get constant volatility
        @param time_index:
        @param spot_i:
        @return:
        '''
        return self.volatility

    def get_simulation_drift(self, time_index):
        '''
        # Get constant drift
        @param time_index:
        @return:
        '''
        return self.drift

class LocalVolatilityHestonModel(SpotDiffusion):
    """
    Local volatility model "equivalent" to Heston model (in the sense of Gyongy theorem).
    """

    def __init__(self, vol_histo, V0, kappa, theta, eta, rho, spot, drift, time_points, number_paths):
        # Calibrate local volatility model from Heston parameters
    
        self.V0 = V0
        self.local_vol_parameters = list(map(lambda t: compute_equiv_local_vol_parameters(v0, kappa, theta, eta, rho, time_points), time_points))
        self.theta = theta

        # Other parameters
        self.rho = rho
        self.spot = spot
        self.drift = drift
        self.time_points = time_points
        self.number_paths = number_paths
        self.vol_histo = vol_histo
        self.v_simulations = np.zeros((len(time_points), self.number_paths), np.float64)

        self.implied_vol_parameters = list(
            map(lambda t: compute_implied_vol_parameters(V0, kappa, theta, eta, rho, time_points[-1]), time_points))

    
    def get_simulation_volatility(self, time_index, spot_i):
        """
        Compute local volatility from (alpha, beta) parameters
        @param time_index:
        @param spot_i:
        @return:
        """
        if time_index == 0:
            sigma_square = smoothed_maximum(self.V0, self.theta * np.pi / 4.0)
        else:
            params = self.local_vol_parameters_list[time_index]
            alpha, beta, gamma = params[0], params[1], params[2]
            sigma_square = smoothed_maximum(alpha + beta * np.log(spot_i / self.spot), self.theta * np.pi / 4.0)

        return np.minimum(np.sqrt(sigma_square), 2.0)

    def get_simulation_drift(self, time_index):
        """
        Get constant drift
        @param time_index:
        @return:
        """
        return self.drift

    def generate_implied_volatility_paths(self, spot_matrix, strike):
        """Generate implied volatility diffusion cone"""
        vol_impl_df = pd.DataFrame(index=spot_matrix.index, columns=spot_matrix.columns.values)

        i= 0
        for column_name, column_data in spot_matrix.iteritems():
            vol_impl_df.at[:, column_name] = list(
                map(lambda s: self.get_implied_volatility_simulation(s, strike), column_data))
            i += 1
        return vol_impl_df

    def get_implied_volatility_simulation(self, time_index, spot_t, strike):
        """Compute implied volatility from (A, B) parameters"""
        params = self.implied_vol_parameters_list[time_index]
        A, B = params[0], params[1]
        if (A, B) in (None, None):
            return 0
        sigma_square = np.maximum(0.0, A + B * np.log(spot_t / strike))
        return np.sqrt(sigma_square)

class HestonStochVol:
    # Class to simulate equity spot/vol risk-factors based on Heston model

    spot: float
    drift: float
    time_points: List[float]
    number_paths: float

    def __init__(self, V0, kappa, theta, eta, rho, spot, drift, time_points, number_paths):
        self.V0 = V0
        self.kappa = kappa
        self.theta = theta
        self.eta = eta
        self.rho = rho
        self.spot = spot
        self.drift = drift
        self.time_points = time_points
        self.number_paths = number_paths
        self.v_simulations = np.zeros((len(self.time_points), self.number_paths), np.float64)

    def get_simulation_volatility(self, time_index, v_i, dt, rand):
        """
        Compute volatility based on Euler simulation scheme with "Full Truncation"
        @param time_index:
        @param v_i:
        @param rand:
        @param dt:
        @return:
        """
        vol_i_plus = np.where(v_i < 0, 0, v_i)
        vol_next = v_i + self.kappa * (self.theta - vol_i_plus) * dt + self.eta * np.sqrt(vol_i_plus) * np.sqrt(dt) * rand
        return vol_next

    def get_simulation_drift(self, time_index):
        """
        Get constant drift
        @param time_index:
        @return:
        """
        return self.drift
    
    def generate_paths(self, scheme="euler"):
        '''
        Generate spot simulations matrix
        @param scheme:
        @return:
        '''
        spot_simulations = np.zeros((len(self.time_points), self.number_paths), np.float64)
        spot_simulations[0] = self.spot

        v_simulations = np.zeros((len(self.time_points), self.number_paths), np.float64)
        v_simulations[0] = self.V0

        for i in range(1, len(self.time_points)):
            #Generate correlated random var
            # z = np.random.normal(size=(2, self.number_paths))
            mean = np.array([0, 0])
            cov = np.array([[1, self.rho], [self.rho, 1]])
            rand = np.random.multivariate_normal(mean, cov, size=self.number_paths)

            dt = (self.time_points[i] - self.time_points[i - 1])
            drift = self.get_simulation_drift(i-1)

            # prev_u_simulations = v_simulations[i - 1] * z[:, 0] * np.sqrt(dt)
            # v_simulations[i] = self.get_simulation_volatility(i, v_simulations[i - 1], z[:, 1], dt)

            prev_v_plus = np.where(v_simulations[i-1] < 0, 0, v_simulations[i-1])

            # Use euler simulation scheme for spot
            if scheme == "euler":
                spot_simulations[i] = euler_scheme(spot_simulations[i - 1], drift, np.sqrt(prev_v_plus), dt, rand[:,0])

            # else use doleans-dade exponential scheme for spot
            else:
                spot_simulations[i] = expo_scheme(spot_simulations[i - 1], drift, np.sqrt(prev_v_plus), dt, rand[:,0])

        # Generate volatility paths based on euler scheme
        v_simulations[i] = self.get_simulation_volatility(i, v_simulations[i - 1], dt, rand[:, 1])

        self.v_simulations = v_simulations
        result = pd.DataFrame(spot_simulations.transpose())
        result.columns = self.time_points

        return result

    def get_v_inst_simulations(self):
        return pd.DataFrame(self.v_simulations.transpose())

# name and main
if __name__ == "__main__":
    directory = r"C:\Users\d52986\Docs\CC\Local Vol Model\Define floor\Spot and variance cones\US_SPX"
    today = datetime.date(2022, 12, 30)
    underlying = "EU STOXX50E"
    dt = 1/260
    n_paths = 10000 # number paths
    time_points = [i * 260.0 for i in range(0, 261)] # time_points (fraction of year)
    diffusion_scheme = "euler"

    # Models parameters

    # Heston model parameters (already calibrated)
    kappa = 4.5822852918317
    theta = 0.5872582875472196
    eta = 1.322328521618614
    rho = -0.7728976081862

    sigma = np.sqrt(theta)
    
    start_date = (today - BDay(1)).date()
    # spot_hts, short_term_varswap_hts = get_spot_varswap_h(ms, from_tss[underlying], ['1'],
    #                                      start_date,
    #                                      today)

    # S0 = spot_ms.loc[np.datetime64(today)]

    # one month warp stats
    # one_month_warps_hp = short_term_warps_hp(short_term_warps_hp.columns.values[0])
    # v0 = get_instantaneous_variance(one_month_warps_hp, kappa, theta, lambda_)

    # Constant volatility model

    # Get bs paths
    constant_vol_model = ConstantVolatilityModel(S0, sigma, time_points, n)
    bs_paths = constant_vol_model.generate_paths(seed=seed, scheme='euler')

    # Calculate realized volatility on each path
    realised_vol_bs = get_realised_volatility(bs_paths)
    accumulated_realised_vol_bs = get_accumulated_realised_volatility_stats(bs_paths)
    accumulated_realised_vol_bs.to_csv(os.path.join(directory, "bs_stats_accumulated_variance.csv"))

    print(f"Average realised volatility: {np.mean(realised_vol_bs):.4f} / Std deviation of realised vol: {np.std(realised_vol_bs):.4f}")
    print(f"Min realised volatility: {np.min(realised_vol_bs):.4f} / Max realised vol: {np.max(realised_vol_bs):.4f}")

    # Plot simulation trajectories
    plot_simulation_paths(bs_paths, directory, "trajectories_bs.png")
    
    # Verify martingality
    stats_spot, stats_spot_maturity = get_stats_spot_distribution(b_paths)
    stats_spot.to_csv(os.path.join(directory, 'b_stats_spot_per_date.csv'))

    # Diff
    b_paths = diff_paths(b_paths[periods>0,:axis].dropna(axis=1))
    stats_spot = get_stats_spot_distribution(b_paths)
    stats_spot.to_csv(os.path.join(directory, 'b_stats_returns_per_date.csv'))

    # Local volatility model

    # Generate spot paths with local vol model
    local_vol_model = LocalVolatilityDiffusion(s0, v0, kappa, theta, eta, rho, sigma_m, time_points, N)
    vol_paths = local_vol_model.generate_paths(scheme='diffusion_scheme')

    # Implied vol approximation with local vol model
    vol_paths = local_vol_model.generate_implied_volatility_paths(vol_paths, s0)
    implied_vol_paths = local_vol_model.generate_implied_volatility_paths(s0)

    stats_spot = get_stats_spot_distribution(vol_paths)
    stats_spot.to_csv(os.path.join(directory, 'lv_stats_spot.csv'))

    plot_simulation_paths(vol_paths, directory, n_trajectories, 'implied_vol_lv.png')

    # Calculate realised volatility on each path
    vol_local = np.std(np.diff(vol_paths)/vol_paths[:,:-1], axis=1)
    vol_local_mean = np.mean(realised_volatility_stats(vol_paths))

    vol_local_std = np.std(realised_volatility_stats(vol_paths))

    stats_spot = get_stats_spot_distribution(vol_local_stats_accumulated_variance(s))

    # Verify realised volatility: (np.mean(realised_vol_local) / std deviation of realised vol: (np.std(realised_vol_local): 4f)
    
    plot_simulation_paths(vol_paths, directory, f"trajectories_loc_vol.png")

    # Verify martingality
    stats_spot_vol = get_stats_maturity / get_stats_spot_distribution(local_vol_paths)
    stats_spot_vol.to_csv(os.path.join(directory, f"rv_stats_spot_per_date.csv"))

    # Diff
    diff_paths = local_vol_paths.diff(periods=1, axis=1).dropna(axis=1)
    stats_returns_lv = get_stats_maturity / get_stats_spot_distribution(diff_paths)
    stats_returns_lv.to_csv(os.path.join(directory, f"rv_stats_returns_per_date.csv"))

    # --- Stochastic volatility with Heston model ---

    # Generate spot paths with stoch vol (Heston model)
    stoch_vol_model = HestonStochVol(kappa, theta, eta, rho, S0, mu, time_points, N)
    spot_vol_paths = stoch_vol_model.generate_paths(stoch_vol_scheme)

    # Calculate realised volatility on each path
    realised_vol = get_realised_volatility(spot_vol_paths)
    accumulated_rv = get_accumulated_realised_volatility(realised_vol)
    stats_accumulated_rv = get_stats_maturity / get_stats_spot_distribution(accumulated_rv)
    stats_accumulated_rv.to_csv(os.path.join(directory, f"stats_accumulated_rv.csv"))

    # Average realised volatility: np.mean(realised_stoch_vol) ; std deviation of realised vol: np.std(realised_stoch_vol) ; df
    df = pd.DataFrame({"avg_rv": np.mean(realised_stoch_vol), "std_rv": np.std(realised_stoch_vol)})
    df.to_csv(os.path.join(directory, f"avg_max_realised_vol.csv"))

    plot_simulation_paths(spot_vol_paths, directory, f"trajectories_stoch_vol.png")

    # Verify martingality
    stats_spot_vol = get_stats_maturity / get_stats_spot_distribution(spot_vol_paths)
    stats_spot_vol.to_csv(os.path.join(directory, f"rv_stats_spot_per_date.csv"))
    
    # Calculate realised volatility of each path
    realised_stoch_vol = get_realised_volatility(stoch_vol_paths)
    realised_svi_vol = get_realised_volatility(svi_vol_paths)
    max_realised_stoch_vol = pd.read_csv("max_realised_volatility_stats_stoch.csv")
    max_realised_svi_vol = pd.read_csv("max_realised_volatility_stats_svi.csv")
    max_realised_stoch_vol = max_realised_stoch_vol["max_realised_stoch_vol"].values
    max_realised_svi_vol = max_realised_svi_vol["max_realised_svi_vol"].values
    print("Average realised volatility: ", np.mean(realised_stoch_vol), " / Std deviation of realised vol: ", np.std(realised_stoch_vol), " / Max realised vol: ", np.max(realised_stoch_vol), " / Max realised vol (from csv): ", max_realised_stoch_vol)

    # Plot spot trajectories
    plot_simulation_output(stoch_vol_paths, directory + "trajectories_stoch_vol.png")

    # Verify martingality
    spot_stoch_vol, spot_svi, spot_maturity = get_stats_spot_distribution(stoch_vol_paths)
    spot_stoch_vol_paths = spot_stoch_vol_paths.join(pd.DataFrame(spot_stoch_vol_paths))
    spot_svi_paths = spot_svi_paths.join(pd.DataFrame(spot_svi_paths))
    spot_maturity = spot_maturity.join(pd.DataFrame(spot_maturity))

    # Diff
    spot_stoch_vol_paths_diff = spot_stoch_vol_paths.diff(axis=1).dropna(axis=1)
    spot_svi_paths_diff = spot_svi_paths.diff(axis=1).dropna(axis=1)
    stats_returns_svi = get_stats_returns_distribution(spot_svi_paths_diff)
    stats_returns_stoch_vol = get_stats_returns_distribution(spot_stoch_vol_paths_diff)
    plot_simulation_output(stats_returns_svi, directory + "SVI_stats_returns_per_date.csv")

    # Compare the spot distributions obtained
    spot_distribution_maturity = pd.concat([spot_svi_maturity, spot_svi_maturity, spot_svi_maturity], axis=1)
    spot_distribution_maturity.columns = ['LV', 'SVI', 'SV']

    sns.histplot(spot_distribution_maturity, kde=True, bins=30, label='variable')
    plt.xlabel('Spot price')
    plt.ylabel('Density')
    plt.savefig(os.path.join(directory, "spot_distribution_hist_mone_chart.png"))
    plt.close()

    sns.displot(spot_distribution_maturity.hist(bins=30, density=True))
    plt.xlabel('Spot price')
    plt.ylabel('Density')
    plt.savefig(os.path.join(directory, "spot_distribution_hist.png"))
    plt.close()






