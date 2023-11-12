import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from scipy import optimize, stats
from scipy.stats import norm


def main():
    random.seed(1234)
    pd.set_option('display.max_columns', None)
    np.set_printoptions(suppress=True)
    # ########## SV dataset ##########
    SV_df = pd.read_csv("SvData.csv")
    Z_t, R_t = 1, 1
    # demeaned log returns
    y_t = SV_df["GBPUSD"]/100
    x_t = np.log((y_t - np.mean(y_t))**2)
    print("*" * 10 + "SvDATA" + "*" * 10)
    plot_stat_time_series(SV_df, y_t, x_t, "SV")

    # MLE
    Parmas = get_initial_parms(x_t)
    c_t, var_eta, T_t, xi, LL = MLE_Partc(Parmas, x_t)
    print("QML estimates for SV data")
    print("c_t = {}, var_eta = {}, T_t = {}, xi = {}, LL = {}".format(np.round(c_t, 4), np.round(var_eta, 4), np.round(T_t, 4), np.round(xi, 4), np.round(LL, 4)))
    a_t, p_t, K_t, F_t, v_t, Filter_H_t = Kalmman_filter(x_t, var_eta, c_t, T_t, xi)
    alpha_t, N_t, r_t = Kalmman_filter_smoother(SV_df, x_t, a_t, p_t, Z_t, T_t, K_t, F_t, v_t, Filter_H_t, xi, "SV", False)

    # (f) Particle filtering(bootstrap filter) - SV data
    bootstrap_resample(xi, SV_df, y_t, var_eta, T_t, Filter_H_t, "SV")


    # (e) - PART 1: repeat analysis for SP500
    ########## SP500 ##########
    SP500_DF = pd.read_csv("SP500.csv")
    SP500_DF["log_RV"] = np.log(SP500_DF["rk_parzen"])
    SP500_DF['log_price'] = np.log(SP500_DF['close_price'])
    n = len(SP500_DF)
    SP_returns = np.zeros(n)
    for i in range(1, n):
        SP_returns[i] = (SP500_DF['log_price'][i] - SP500_DF['log_price'][i-1])
    SP500_DF["Return"] = SP_returns
    SP500_DF = SP500_DF[1:].reset_index(drop=True) # excluding the first obs
    y_t = SP500_DF["Return"]
    x_t = np.log((y_t - np.mean(y_t))**2)
    print("*" * 10 + "SP500" + "*" * 10)
    plot_stat_time_series(SP500_DF, y_t, x_t, "SP500")

    # MLE
    Parmas = get_initial_parms(x_t)
    c_t, var_eta, T_t, xi, LL = MLE_Partc(Parmas, x_t)
    print("QML estimates for SP500 data")
    print("c_t = {}, var_eta = {}, T_t = {}, xi = {}, LL = {}".format(np.round(c_t, 4), np.round(var_eta, 4), np.round(T_t, 4), np.round(xi, 4),  np.round(LL, 4)))
    a_t, p_t, K_t, F_t, v_t, Filter_H_t = Kalmman_filter(x_t, var_eta, c_t, T_t, xi)
    alpha_t, N_t, r_t = Kalmman_filter_smoother(SP500_DF, x_t, a_t, p_t, Z_t, T_t, K_t, F_t, v_t, Filter_H_t, xi, "SP500", False)

    # (e) - PART 2: Inclusion of log(RV). Compute GLS estimate for beta
    prediction_err = Kalmman_filter_RV(var_eta, c_t, T_t, SP500_DF["log_RV"])
    beta_est = np.sum(prediction_err*v_t/F_t)/np.sum(prediction_err**2/F_t)
    beta_var = 1/np.sum(prediction_err**2/F_t)
    print("Estimate of beta:", np.round(beta_est, 4))
    print("Variance of beta: ", np.round(beta_var, 4))

    # (f) Particle filtering(bootstrap filter) - SP500 data
    bootstrap_resample(xi, SP500_DF, y_t, var_eta, T_t, Filter_H_t, "SP500")

    # MLE with new x_t
    new_xt = x_t - SP500_DF["log_RV"]*beta_est
    Parmas = get_initial_parms(new_xt)
    c_t, var_eta, T_t, xi, LL = MLE_Partc(Parmas, new_xt)
    print("QML estimates for new SP500 data")
    print("c_t = {}, var_eta = {}, T_t = {}, xi = {}, LL = {}".format(np.round(c_t, 4), np.round(var_eta, 4), np.round(T_t, 4),
                                                             np.round(xi, 4), np.round(LL, 4)))
    a_t, p_t, K_t, F_t, v_t, Filter_H_t = Kalmman_filter(new_xt, var_eta, c_t, T_t, xi)
    alpha_t, N_t, r_t = Kalmman_filter_smoother(SP500_DF, new_xt, a_t, p_t, Z_t, T_t, K_t, F_t, v_t, Filter_H_t, xi,"SP500", True)

def bootstrap_resample(xi, df, y_t, var_eta, phi, Filer_H_t, type):
    N = 10000
    T = len(y_t)
    alpha = np.zeros((N, T))
    weights = np.zeros((N, T))
    normalized_w = np.zeros((N, T))
    filterd_a = np.zeros(T)
    sigma_t = np.zeros((N, T))
    for t in range(T):
        if t == 0:
            uncond_mean_rec = 0
            uncond_sig_rec = np.sqrt(var_eta / (1 - phi ** 2))
        else:
            uncond_mean_rec = phi*alpha[:, t-1] # this is updated for each iteration after line 102 (resampling)
            uncond_sig_rec = np.sqrt(var_eta)
        alpha[:, t] = np.random.normal(uncond_mean_rec, uncond_sig_rec, N)
        sigma_t[:, t] = np.sqrt(np.exp(xi+alpha[:, t]))
        # More general form than providing the likelihood function
        weights[:, t] = norm.pdf(y_t[t], np.mean(y_t), scale=sigma_t[:, t])
        normalized_w[:, t] = weights[:, t]/ np.sum(weights[:, t])
        filterd_a[t] = np.sum(normalized_w[:, t] * alpha[:, t])
        # resampling
        alpha[:, t] = np.random.choice(alpha[:, t], size = N, replace=True, p=normalized_w[:, t])

    if type == "SP500":
        obs_index = pd.to_datetime(df["Date"], format='%Y%m%d')
    else:
        obs_index = df.index

    plt.rcParams["figure.figsize"] = [16, 10]
    plt.plot(obs_index, filterd_a, c='red', alpha=0.5, lw=1, label="Particle filter: H_t")
    plt.plot(obs_index, Filer_H_t, c='blue', alpha=0.5, lw=1, label="Filtered QML: H_t")
    plt.legend(prop={'size': 10})
    plt.title("Comparison between filtered H_t using Particle filter method and QML method for " + type,  size = 20)
    plt.show()


def get_initial_parms(x_t):
    d_t = -1.27
    H_t = (math.pi)**2/2
    mu_x = np.mean(x_t)
    var_x = np.var(x_t)
    T_t_ini = 0.9
    var_eta_ini = (var_x - H_t)*(1-T_t_ini**2)
    c_t_ini = (mu_x - d_t)*(1-T_t_ini)
    Parmas = np.array([c_t_ini, var_eta_ini, T_t_ini])
    return Parmas


def MLE_Partc(Parmas, x_t):
    bnds = ((None, None), (0, None),  (0, 0.9999))
    opt = {'disp': True, 'maxiter' : 5000, 'eps': 1e-06}
    llik_diffuse = lambda y: (diffuse_MLE(y, x_t)[0])
    results_MLE = scipy.optimize.minimize(llik_diffuse, Parmas, bounds=bnds, method='SLSQP') # L-BFGS-B
    c_t, var_eta, T_t = results_MLE.x[0], results_MLE.x[1], results_MLE.x[2]
    xi = c_t / (1 - T_t)
    log_like = diffuse_MLE(results_MLE.x, x_t)[1]
    return c_t, var_eta, T_t, xi, log_like


def simulation_MLE(pars, x_t, d_t):
    n = len(x_t)
    c_t_est, var_eta_est, T_t_est = pars[0], pars[1], pars[2]
    vEps = np.random.randn(n) * np.sqrt((math.pi) ** 2 / 2)
    vEta = np.random.randn(n) * np.sqrt(var_eta_est)
    h_t = np.zeros(n + 1)
    h_t[0] = c_t_est/(1-T_t_est)
    x_t_sim = np.zeros(n)
    for i in range(n):
        h_t[i + 1] = T_t_est * h_t[i] + c_t_est + vEta[i]
        x_t_sim[i] = h_t[i] + d_t + vEps[i]
    return x_t_sim

def Kalmman_filter(x_t, var_eta, c_t, T_t, xi):
    Z_t, R_t = 1, 1
    d_t = -1.27
    H_t = (math.pi)**2/2
    n = len(x_t)
    v_t, Filter_a_t, Filter_p_t, M_t, K_t, F_t, Filer_a_t, Pred_a_t, Filer_p_t, Pred_p_t = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n + 1), np.zeros(n), np.zeros(n + 1),
    Pred_a_t[0], Pred_p_t[0] = 0, 10 ** 7
    for i in range(n):
        v_t[i] = x_t[i] - Z_t * Pred_a_t[i] - d_t
        F_t[i] = Z_t * Pred_p_t[i] * Z_t + H_t
        K_t[i] = T_t * Pred_p_t[i] * Z_t / F_t[i]
        M_t[i] = Pred_p_t[i] * Z_t / F_t[i]
        Filer_a_t[i] = Pred_a_t[i] + M_t[i] * v_t[i]
        Filer_p_t[i] = Pred_p_t[i] - M_t[i] * F_t[i] * M_t[i]
        Pred_a_t[i + 1] = T_t * Filer_a_t[i] + c_t
        Pred_p_t[i + 1] = T_t * Filer_p_t[i] * T_t + R_t * (var_eta) * R_t
    Filer_H_t = Filer_a_t - xi
    a_t, P_t = Pred_a_t[:-1], Pred_p_t[:-1]
    return a_t, P_t, K_t, F_t, v_t, Filer_H_t


def Kalmman_filter_RV(var_eta, c_t, T_t, log_RV):
    Z_t, R_t = 1, 1
    #d_t = -1.27
    H_t = (math.pi)**2/2
    n = len(log_RV)
    v_t, Filter_a_t, Filter_p_t, M_t, K_t, F_t, Filer_a_t, Pred_a_t, Filer_p_t, Pred_p_t = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n + 1), np.zeros(n), np.zeros(n + 1),
    Pred_a_t[0], Pred_p_t[0] = 0, 10 ** 7
    for i in range(n):
        v_t[i] = log_RV[i] - Z_t * Pred_a_t[i]
        F_t[i] = Z_t * Pred_p_t[i] * Z_t + H_t
        K_t[i] = T_t * Pred_p_t[i] * Z_t / F_t[i]
        M_t[i] = Pred_p_t[i] * Z_t / F_t[i]
        Filer_a_t[i] = Pred_a_t[i] + M_t[i] * v_t[i]
        Filer_p_t[i] = Pred_p_t[i] - M_t[i] * F_t[i] * M_t[i]
        Pred_a_t[i + 1] = T_t * Filer_a_t[i] + c_t
        Pred_p_t[i + 1] = T_t * Filer_p_t[i] * T_t + R_t * (var_eta) * R_t
    return v_t

def Kalmman_filter_smoother(df, x_t, a_t, P_t, Z_t, T_t, K_t, F_t, v_t, Filer_H_t, xi, series, new_x):
    n = len(x_t)
    alpha_t, V_t, r_t, N_t, L_t = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    # initialization of parameters
    r_t[-1], N_t[-1] = 0, 0
    for i in range(n-1, -1, -1):
        L_t[i] = T_t - K_t[i]*Z_t
        r_t[i-1] = Z_t/F_t[i]*v_t[i]+L_t[i]*r_t[i]
        N_t[i-1] = Z_t/F_t[i]*Z_t + L_t[i]*N_t[i]*L_t[i]
        alpha_t[i] = a_t[i] + P_t[i]*r_t[i-1]
        V_t[i] = P_t[i] - P_t[i]*N_t[i-1]*P_t[i]
    if series == "SP500":
        obs_index = pd.to_datetime(df["Date"], format='%Y%m%d')
    else:
        obs_index = df.index
    fig, ax = plt.subplots(2, 1, figsize=(16, 10))
    smoothed_H_t = alpha_t - xi
    ax[0].scatter(obs_index, x_t, s = 2, label = "x_t")
    ax[0].plot(obs_index, alpha_t, c = 'red', label = "smoothed h_t")
    if new_x:
        ax[0].set_title('(i):' + 'new_x: x_t - beta*log(RV)' + 'and smoothed h_t')
    else:
        ax[0].set_title('(i): x_t and smoothed h_t')
    ax[1].plot(obs_index, Filer_H_t, label = "Filtered H_t")
    ax[1].plot(obs_index, smoothed_H_t, c = 'red', label = "Smoothed H_t")
    ax[1].set_title('(ii): Filtered H_t and Smoothed H_t')
    plt.legend(prop={'size': 10})
    plt.show()

    return alpha_t, N_t, r_t


def diffuse_MLE(Parmas, x_t):
    Z_t, R_t = 1, 1
    d_t = -1.27
    H_t = (math.pi)**2/2
    # d_t is known = -1.27, and c_t is to be estimated
    c_t = Parmas[0]
    var_eta = Parmas[1] # Q_t
    T_t = Parmas[2]
    n = len(x_t)
    v_t, Filter_a_t, Filter_p_t, F_t, K_t, Filer_a_t, Pred_a_t, Filer_p_t, Pred_p_t = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n+1), np.zeros(n), np.zeros(n+1),
    # initialization of parameters
    Pred_a_t[0] = c_t/(1-T_t)
    Pred_p_t[0] = var_eta/(1-T_t**2)
    for i in range(n):
        v_t[i] = x_t[i] - Z_t * Pred_a_t[i] - d_t
        F_t[i] = Z_t*Pred_p_t[i]*Z_t + H_t
        K_t[i] = T_t * Pred_p_t[i] / F_t[i]
        # Pred_a_t[i + 1] = c_t + T_t*Pred_a_t[i] + K_t[i] * v_t[i]
        # Pred_p_t[i + 1] = (T_t**2) * Pred_p_t[i] + (var_eta) - (K_t[i]**2)*F_t[i]
        Filer_a_t[i] = Pred_a_t[i] + Pred_p_t[i]*Z_t/F_t[i]*v_t[i]
        Filer_p_t[i] = Pred_p_t[i] - Pred_p_t[i]*Z_t/F_t[i]*Z_t*Pred_p_t[i]
        Pred_a_t[i+1] = T_t*Filer_a_t[i] + c_t
        Pred_p_t[i+1] = T_t*Filer_p_t[i]*T_t + R_t*var_eta*R_t

    LL = -n/2 * np.log(2 * math.pi) - 1 / 2 * np.sum(np.log(F_t)+v_t**2/F_t)
    dLL = -LL/n
    return dLL, LL

def plot_stat_time_series(df, y_t, x_t, type):
    if type == "SP500":
        obs_index = pd.to_datetime(df["Date"], format='%Y%m%d')
    else:
        obs_index = df.index
    fig, ax = plt.subplots(2, 1, figsize=(16, 10))
    hline = np.zeros(len(y_t))
    ax[0].plot(obs_index, y_t)
    ax[0].plot(obs_index, hline)
    ax[0].set_title('(i): Daily log returns: y_t')
    ax[1].plot(obs_index, x_t)
    ax[1].set_title('(ii): Log of demeaned returns: x_t')
    plt.show()
    print("Descriptive Statistics of y_t: " )
    print(y_t.describe())
    print("Kurtosis = {} and Skewness = {}\n".format(np.round(y_t.kurtosis(), 4), np.round(y_t.skew(), 4)))
    print("Descriptive Statistics of x_t: " )
    print(x_t.describe())
    print("Kurtosis = {} and Skewness = {}\n".format(np.round(x_t.kurtosis(), 4), np.round(x_t.skew(), 4)))
    if type == "SP500":
        plt.rcParams["figure.figsize"] = [16, 10]
        plt.plot(obs_index, df["log_RV"], c='red', alpha= 0.5, lw=1, label = "log(RV): explanatory variable")
        plt.plot(obs_index, x_t, c='green', alpha= 0.5, lw=1, label = "x_t: Observation")
        plt.legend(prop={'size': 10})
        plt.title("Time series of RV_Parzen vs x_t",  size = 20)
        plt.show()

if __name__ == "__main__":
    main()


