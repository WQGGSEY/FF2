import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import os
from numba import jit, njit, prange
from tqdm import tqdm, trange

SEED = 42
np.random.seed(SEED) # for reproducibility


def construct_X_and_Y(file_path='simulation_data')->tuple[np.ndarray, np.ndarray]:
    """
    Parses the data from a file where each line contains two space-separated floats.    
    """

    start_date = '2016-03-30'
    end_date = '2025-03-20'
    with open(f"{file_path}/KS200_MASK.pkl", 'rb') as f:
        mask_df:pd.DataFrame = pickle.load(f).ffill(axis=1)
        mask_df = mask_df.loc[:, start_date:end_date]
    with open(f"{file_path}/Return.pkl", 'rb') as f:
        returns_df = pickle.load(f).ffill(axis=1) * 0.01
        returns_df = returns_df.loc[:, start_date:end_date]
    with open(f"{file_path}/MarketCap.pkl", 'rb') as f:
        mc_df = pickle.load(f).ffill(axis=1)
        mc_df = mc_df.loc[:, start_date:end_date]
    with open(f"{file_path}/ifrs-full_Equity.pkl", 'rb') as f:
        be_df = pickle.load(f).ffill(axis=1)
        be_df = be_df.loc[:, start_date:end_date]
    with open(f"{file_path}/KOSPI_Close.pkl", 'rb') as f:
        kospi_close_df = pickle.load(f).ffill(axis=1)
        kospi_return_df = kospi_close_df.pct_change(axis=1)
        mkt_df = kospi_return_df.loc[:, start_date:end_date]
    with open(f"{file_path}/rf_bond.pkl", 'rb') as f:
        rf_df = pickle.load(f).ffill(axis=1) * 0.01
        rf_df = rf_df.loc[:, start_date:end_date]
    with open(f"{file_path}/corp_aa_bond.pkl", 'rb') as f:
        corp_aa_df = pickle.load(f).ffill(axis=1) * 0.01
        corp_aa_df = corp_aa_df.loc[:, start_date:end_date]
    with open(f"{file_path}/corp_bb_bond.pkl", 'rb') as f:
        corp_bb_df = pickle.load(f).ffill(axis=1) * 0.01
        corp_bb_df = corp_bb_df.loc[:, start_date:end_date]
    with open(f"{file_path}/gov10_bond.pkl", 'rb') as f:
        gov10_df = pickle.load(f).ffill(axis=1) * 0.01
        gov10_df = gov10_df.loc[:, start_date:end_date]
    with open(f"{file_path}/gov3_bond.pkl", 'rb') as f:
        gov3_df = pickle.load(f).ffill(axis=1) * 0.01
        gov3_df = gov3_df.loc[:, start_date:end_date]
    corp_df = (corp_aa_df + corp_bb_df) / 2
    R     = returns_df
    MC    = mc_df
    BE    = be_df 
    RF = rf_df / 252 # 일단위 reblancing 시행하므로 252로 나누어 줌.
    GOV10   = gov10_df / 252
    GOV3 = gov3_df / 252
    CORP  = corp_df / 252
    MKT   = mkt_df
    mask  = mask_df.astype(bool)

    dates = R.columns
    
    port_rets = {f"S{i}B{j}":[] for i in range(1,6) for j in range(1,6)}

    for dt in dates:
        members = mask[dt][mask[dt]].index
        if members.empty: 
            continue

        # size quintile breakpoints
        me = MC.loc[members, dt].dropna()
        sz_br = me.quantile([0.2,0.4,0.6,0.8]).values

        # BE/ME quintile breakpoints
        bm = (BE.loc[members, dt] / MC.loc[members, dt]).dropna()
        bm_br = bm.quantile([0.2,0.4,0.6,0.8]).values

        # 각 포트폴리오
        for i in range(1,6):
            size_idx = me[
                (me >  (sz_br[i-2] if i>1 else -np.inf)) &
                (me <= (sz_br[i-1] if i<5 else  np.inf))
            ].index
            for j in range(1,6):
                bm_idx = bm[
                (bm >  (bm_br[j-2] if j>1 else -np.inf)) &
                (bm <= (bm_br[j-1] if j<5 else  np.inf))
                ].index

                idx = size_idx.intersection(bm_idx)
                name = f"S{i}B{j}"
                if idx.empty:
                    ret = np.nan
                else:
                    w = MC.loc[idx, dt]
                    w = w / w.sum()
                    ret = (R.loc[idx, dt] * w).sum()
                port_rets[name].append((dt, ret))

    # DataFrame으로 변환
    port_df = pd.DataFrame({
        name: pd.Series(dict(port_rets[name]))
        for name in port_rets
    }).sort_index(axis=1)
    port_df = port_df.interpolate(method='linear', axis=0, limit_direction='both')


    # ── 2) SMB, HML 계산 ──
    # SMB: size quintile 1 평균 – quintile 5 평균
    SMB = (port_df[[f"S1B{j}" for j in range(1,6)]].mean(axis=1) + port_df[[f"S2B{j}" for j in range(1,6)]].mean(axis=1)) \
            - (port_df[[f"S5B{j}" for j in range(1,6)]].mean(axis=1) + port_df[[f"S4B{j}" for j in range(1,6)]].mean(axis=1)) 

    # HML: BE quintile 5 평균 – quintile 1 평균
    HML = (port_df[[f"S{i}B5" for i in range(1,6)]].mean(axis=1) + port_df[[f"S{i}B4" for i in range(1,6)]].mean(axis=1))\
        - (port_df[[f"S{i}B1" for i in range(1,6)]].mean(axis=1) + port_df[[f"S{i}B2" for i in range(1,6)]].mean(axis=1))


    factors = pd.DataFrame({
        'MKT_RF': MKT.iloc[0] - RF.iloc[0],
        'SMB':     SMB,
        'HML':     HML,
        'TERM':    GOV10.iloc[0]  - RF.iloc[0],
        'DEF':     CORP.iloc[0] - GOV3.iloc[0],
        'RF':      RF.iloc[0]
    }, index=dates).dropna()
    factor_cols = ['MKT_RF', 'SMB', 'HML', 'TERM', 'DEF'] # drop 'RF' as it is not needed for regression
    X = np.c_[np.ones(len(factors)), factors[factor_cols].to_numpy()] # X: T x 6 = 2203 x 6
    Y = port_df.subtract(factors['RF'], axis=0).to_numpy() # Y: T x 25 = 2203 x 25
    return X, Y

@jit(nopython=True)
def sample_posterior_beta(y: np.ndarray, X: np.ndarray, sigma_sq: float, g: int):
    """
    Sample from the posterior distribution of beta given y and X.
    y: nx1 (n=532)
    X: nxp (p=7)
    g: int (g-prior)
    """
    y = np.ascontiguousarray(y)
    y = y.reshape(-1, 1)
    mvn_mean = g/(g+1) * np.linalg.pinv(X.T @ X) @ X.T @ y
    mvn_cov = g/(g+1) * sigma_sq * np.linalg.pinv(X.T @ X)

    L = np.linalg.cholesky(mvn_cov)
    z = np.random.randn(X.shape[1])
    return (mvn_mean.flatten() + L @ z)

@jit(nopython=True)
def sample_posterior_sigma(y: np.ndarray, X: np.ndarray, nu_0: float, sigma_0_sq: float, g: int):
    """
    Sample from the posterior distribution of sigma given Y and X.
    """
    y = np.ascontiguousarray(y)
    y = y.reshape(-1, 1)
    a = (nu_0 + y.shape[0])/2
    SSR_g = y.T @ (np.eye(y.shape[0]) - g/(g+1)* X @ np.linalg.pinv(X.T @ X) @ X.T) @ y
    b = (nu_0 * sigma_0_sq + SSR_g)/2
    return b/np.random.gamma(a, scale=1)

def bayesian_regression(file_path:str, nu_0: float, S=5000):
    """
    Note that Pr(beta_j neq 0 | y) = Pr(z_j = 1 | y)
    Model: y = beta_0 + beta_1 * MKT_RF + beta_2 * SMB + beta_3 * HML + beta_4 * TERM + beta_5 * DEF + e
    """
    X, Y = construct_X_and_Y(file_path)
    columns = ['Intercept', 'MKT_RF', 'SMB', 'HML', 'TERM', 'DEF']
    # Data Settings
    g = Y.shape[0]  # g-prior
    # MCMC sampling
    # model space = 2*6 = 64
    # we will use uniform prior for model selection, i.e. random selection
    if os.path.exists(f"{file_path}/mcmc_samples.pkl"):
        with open(f"{file_path}/mcmc_samples.pkl", 'rb') as f:
            portfolio_samples = pickle.load(f)
    else:
        portfolio_samples = []
        count = 1
        for y in tqdm(Y.T, desc=f"Processing each portfolio", leave=False):
            # Initialize z to all zeros (no variables included)
            z_init = np.ones(X.shape[1], dtype=np.int8)
            portfolio_samples += mcmc(y, X, g, z_init, nu_0=nu_0, S=S)
            count += 1
        with open(f"{file_path}/mcmc_samples.pkl", 'wb') as f:
            pickle.dump(portfolio_samples, f)

    index_counts = []
    for samples in portfolio_samples:
        index_count = np.array([0 for _ in range(X.shape[1])])
        for sample in samples:
            z, _, _ = sample
            index_count += z

        index_count = index_count / len(samples)  # Normalize to get probabilities
        # 2. Obtain the Pr(beta_j neq 0 | y) and its 95% CI
        # beta_samples = np.array([sample[2] for sample in samples])  # i: sample index, j: beta index
        print("Posterior probabilities of beta_j != 0 <-> z_j = 1 and their 95% CI")
        print("----------------------------------------------------------------------------------------------------")
        print(f"{"Variable":<10} | {"Posterior Prob":<15} | {"95% CI Lower Bound":<18} | {"95% CI Upper Bound":<18} | {"CI Width":<10}")
        print("----------------------------------------------------------------------------------------------------")
        for i in range(index_count.shape[0]):
            print(f"{columns[i]:<10} | {index_count[i]:<15.4f} | {max(0, index_count[i] - 1.96 * np.sqrt(index_count[i] * (1 - index_count[i]) / len(samples))):<18.4f} | {min(1, index_count[i] + 1.96 * np.sqrt(index_count[i] * (1 - index_count[i]) / len(samples))):<18.4f} | {2 * 1.96 * np.sqrt(index_count[i] * (1 - index_count[i]) / len(samples)):<10.4f}")
        print("----------------------------------------------------------------------------------------------------")
        index_counts.append(index_count.copy())

    # Plotting the posterior probabilities
    rows = ['S1','S2','S3','S4','S5']
    cols = ['B1','B2','B3','B4','B5'] 
    feature_names = ['Intercept', 'MKT_RF', 'SMB', 'HML', 'TERM', 'DEF']

    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(20, 20), sharey=True)

    for i, b in enumerate(rows):
        for j, s in enumerate(cols):
            ax = axes[i, j]
            portfolio_name = f"{s}{b}"
            data = index_counts[i * len(rows) + j]

            ax.bar(feature_names, data, color='skyblue')
            ax.set_title(portfolio_name)
            ax.set_xticklabels(feature_names, rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

        
@jit(nopython=True)
def X_z(X:np.ndarray, z:np.ndarray):
    """
    Returns the design matrix X with only the columns corresponding to z=1.
    X: nxp (p=7)
    z: model space
    Z_z: nxp_z
    drop the column of X if zj=0
    """
    return X[:, z==1]
    
def mcmc(y: np.ndarray, X: np.ndarray, g: int, z_init: np.ndarray, nu_0=1.0, S=5_000):
    """
    MCMC sampling for Bayesian linear regression with model selection
    y: nx1
    X: nxp
    g: int (g-prior)
    model_space: list of tuples, each tuple represents a model
    S: int (number of samples)
    z: model_space
    """
    y = np.ascontiguousarray(y)
    y = y.reshape(-1, 1)  # Ensure y is a column vector
    z = z_init.copy()  # Initialize z
    samples=[]
    for s in trange(S, leave=True):
        # Randomly select a model from the model space
        z = sample_posterior_model(y, X, z, nu_0, g)
        X_zj = X_z(X, z)  # Get the design matrix for the selected model
        sigma0_sq = y.T @ (np.eye(y.shape[0]) - X_zj @ np.linalg.pinv(X_zj.T @ X_zj) @ X_zj.T) @ y / (y.shape[0] - np.linalg.matrix_rank(X_zj))
        sigma_sq = sample_posterior_sigma(y, X_zj, nu_0, sigma0_sq, g)
        beta = sample_posterior_beta(y, X_zj, sigma_sq, g)
        samples.append((z.copy(), sigma_sq, beta))
    return samples

@jit(nopython=True)
def sample_posterior_model(y: np.ndarray, X: np.ndarray, z: np.ndarray, nu_0: float, g: int):
    y = np.ascontiguousarray(y)
    y = y.reshape(-1, 1)
    for j in range(z.shape[0]):
        odds = compute_odds(z, y, X, nu_0, g, j)
        posterior_prob = odds / (1 + odds)
        z[j] = 1 if np.random.rand() < posterior_prob else 0
    return z

@jit(nopython=True)
def compute_odds(z: np.ndarray, y: np.ndarray, X: np.ndarray, nu_0: float, g: int, j: int):
    temp_z = z.copy()
    temp_z[j] = 1
    z_j_1 = temp_z.copy()
    temp_z[j] = 0
    z_j_0 = temp_z.copy()
    X_j_1 = X_z(X, z_j_1)
    X_j_0 = X_z(X, z_j_0)
    p_j_1 = np.linalg.matrix_rank(X_j_1)
    p_j_0 = np.linalg.matrix_rank(X_j_0)
    SSR_g_1 = y.T @ (np.eye(y.shape[0]) - g/(g+1)* X_j_1 @ np.linalg.pinv(X_j_1.T @ X_j_1) @ X_j_1.T) @ y
    SSR_g_0 = y.T @ (np.eye(y.shape[0]) - g/(g+1)* X_j_0 @ np.linalg.pinv(X_j_0.T @ X_j_0) @ X_j_0.T) @ y
    # set sigma_0_sq = s_z^2
    sigma_1_sq = y.T @ (np.eye(y.shape[0]) - X_j_1 @ np.linalg.pinv(X_j_1.T @ X_j_1) @ X_j_1.T) @ y / (y.shape[0] - p_j_1)
    sigma_0_sq = y.T @ (np.eye(y.shape[0]) - X_j_0 @ np.linalg.pinv(X_j_0.T @ X_j_0) @ X_j_0.T) @ y / (y.shape[0] - p_j_0)
    odds = (1+g)**((p_j_0 - p_j_1)/2) * (sigma_1_sq / sigma_0_sq)**(nu_0/2) * ((sigma_0_sq + SSR_g_0) / (sigma_1_sq + SSR_g_1))**((nu_0+y.shape[0])/2)
    return odds
    

def main(file_path, nu_0, S):
    """
    Prediction of stock returns using linear combination of other variables
    """
    bayesian_regression(file_path, nu_0, S)
    
    
    
if __name__ == "__main__":
    if (len(sys.argv) > 4):
        print("Usage: python3 BayesianFF.py [file_path] [nu_0] [S]")
    file_path = sys.argv[1] if len(sys.argv) > 1 else 'simulation_data'
    nu_0 = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    S = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    print(f"File path: {file_path}, nu_0: {nu_0}, S: {S}\n Executing BayesianFF.py ...")
    main(file_path, nu_0, S)