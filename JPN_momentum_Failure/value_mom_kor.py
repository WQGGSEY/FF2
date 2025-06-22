import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import os
from numba import jit, njit, prange
from tqdm import tqdm, trange
import pymc as pm
import pytensor as pt
import arviz as az

######################## DATA PREPARATION ########################

file_path='simulation_data'
start_date = '2016-01-02'
end_date = '2024-12-30'
start_year = int(start_date.split('-')[0])
end_year = int(end_date.split('-')[0])
with open(f"{file_path}/KS200_MASK.pkl", 'rb') as f:
    mask_df:pd.DataFrame = pickle.load(f).ffill(axis=1)
    mask_df = mask_df.loc[:, start_date:end_date]
with open(f"{file_path}/Return.pkl", 'rb') as f:
    returns_df:pd.DataFrame = pickle.load(f).ffill(axis=1) * 0.01
    returns_df = returns_df.loc[:, start_date:end_date]
with open(f"{file_path}/MarketCap.pkl", 'rb') as f:
    mc_df:pd.DataFrame = pickle.load(f).ffill(axis=1)
    mc_df = mc_df.loc[:, start_date:end_date]
with open(f"{file_path}/ifrs-full_Equity.pkl", 'rb') as f:
    be_df:pd.DataFrame = pickle.load(f).ffill(axis=1)
    be_df = be_df.loc[:, start_date:end_date]
with open(f"{file_path}/KOSPI_Close.pkl", 'rb') as f:
    kospi_close_df:pd.DataFrame = pickle.load(f).ffill(axis=1)
    kospi_return_df = kospi_close_df.pct_change(axis=1)
    mkt_df = kospi_return_df.loc[:, start_date:end_date]
with open(f"{file_path}/rf_bond.pkl", 'rb') as f:
    rf_df:pd.DataFrame = pickle.load(f).ffill(axis=1) * 0.01
    rf_df = rf_df.loc[:, start_date:end_date]
with open(f"{file_path}/corp_aa_bond.pkl", 'rb') as f:
    corp_aa_df:pd.DataFrame = pickle.load(f).ffill(axis=1) * 0.01
    corp_aa_df = corp_aa_df.loc[:, start_date:end_date]
with open(f"{file_path}/corp_bb_bond.pkl", 'rb') as f:
    corp_bb_df:pd.DataFrame = pickle.load(f).ffill(axis=1) * 0.01
    corp_bb_df = corp_bb_df.loc[:, start_date:end_date]
with open(f"{file_path}/gov10_bond.pkl", 'rb') as f:
    gov10_df:pd.DataFrame = pickle.load(f).ffill(axis=1) * 0.01
    gov10_df = gov10_df.loc[:, start_date:end_date]
with open(f"{file_path}/gov3_bond.pkl", 'rb') as f:
    gov3_df:pd.DataFrame = pickle.load(f).ffill(axis=1) * 0.01
    gov3_df = gov3_df.loc[:, start_date:end_date]
corp_df = (corp_aa_df + corp_bb_df) / 2


#################################################################
RESULT_PATH = 'JPN_momentum_Failure/results'
if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)
#################################################################


################ HyperParameter For MCMC ########################
SEED = 42
LOAD = True
SAMPLE_SIZE = 2000
TUNE = 1000
chains = 4
target_accept = 0.95

d_a0 = 0.01
d_a1 = 0.01
d_h0 = 0.01
d_h1 = 0.01
d = 0.01
c_0 = 0.01
c_1 = 0.01
T = end_year - start_year + 1  # Total number of years in the dataset
#################################################################



def compute_mom_and_value_return():
    """
    가치(B/M) 및 모멘텀 신호를 기반으로 연간 리밸런싱 포트폴리오를 구성하고,
    Lookahead Bias를 제거하여 실제 투자 가능한 전략의 수익률을 계산합니다.
    """
    annual_return_df_list = []
    for year in range(start_year, end_year + 1):
        temp_return_df = returns_df[returns_df.columns[returns_df.columns.str.startswith(str(year))]]
        annual_return_df = (1 + temp_return_df).prod(axis=1) - 1
        annual_return_df.name = str(year)
        annual_return_df_list.append(annual_return_df)
    annual_return_df = pd.concat(annual_return_df_list, axis=1)

    annual_mask_df_list = []
    for year in range(start_year, end_year + 1):
        temp_mask_df = mask_df[mask_df.columns[mask_df.columns.str.startswith(str(year))]]
        annual_mask_df = temp_mask_df.max(axis=1)
        annual_mask_df.name = str(year)
        annual_mask_df_list.append(annual_mask_df)
    annual_mask_df = pd.concat(annual_mask_df_list, axis=1).fillna(0).astype(int)

    value_df = be_df / mc_df
    annual_value_df_list = []
    for year in range(start_year, end_year + 1):
        temp_value_df = value_df[value_df.columns[value_df.columns.str.startswith(str(year))]]
        if temp_value_df.empty: continue
        annual_value_df = temp_value_df.iloc[:, -1]
        annual_value_df.name = str(year)
        annual_value_df_list.append(annual_value_df)
    annual_value_df = pd.concat(annual_value_df_list, axis=1).fillna(0)
    
    annual_mom_df_list = []
    for year in range(start_year, end_year + 1):
        temp_return_df = returns_df[returns_df.columns[returns_df.columns.str.startswith(str(year))]]
        temp_return_df = temp_return_df.drop(temp_return_df.columns[temp_return_df.columns.str.contains('-12-')], axis=1, errors='ignore')
        annual_mom_df = (1 + temp_return_df).prod(axis=1) - 1
        annual_mom_df.name = str(year)
        annual_mom_df_list.append(annual_mom_df)
    annual_mom_df = pd.concat(annual_mom_df_list, axis=1)

    annual_mc_df_list = []
    for year in range(start_year, end_year + 1):
        temp_mc_df = mc_df[mc_df.columns[mc_df.columns.str.startswith(str(year))]]
        if temp_mc_df.empty: continue
        annual_mc_df = temp_mc_df.iloc[:, -1]
        annual_mc_df.name = str(year)
        annual_mc_df_list.append(annual_mc_df)
    annual_mc_df = pd.concat(annual_mc_df_list, axis=1)

    annual_rf_df = (rf_df.groupby(pd.to_datetime(rf_df.columns).year, axis=1).mean()).transpose().iloc[:, 0]
    annual_rf_df.index = annual_rf_df.index.astype(str)
    annual_rf_df = annual_rf_df.to_frame(name='Risk-Free Rate')

    value_portfolio_df_list = []
    mom_portfolio_df_list = []
    for year in range(start_year, end_year + 1):
        year_str = str(year)
        members = annual_mask_df[year_str][annual_mask_df[year_str]==1].index
        if members.empty: continue

        value = annual_value_df.loc[members, year_str].dropna()
        mom = annual_mom_df.loc[members, year_str].dropna()
        value_brp = value.quantile([0.5]).values
        mom_brp = mom.quantile([0.5]).values
        
        long_value = value[value >= value_brp[0]]
        short_value = value[value < value_brp[0]]
        long_mom = mom[mom >= mom_brp[0]]
        short_mom = mom[mom < mom_brp[0]]

        mc_t = annual_mc_df.loc[:, year_str]
        
        long_value_weighted = mc_t.loc[long_value.index] / mc_t.loc[long_value.index].sum()
        short_value_weighted = -mc_t.loc[short_value.index] / mc_t.loc[short_value.index].sum()
        value_portfolio = pd.concat([long_value_weighted, short_value_weighted])
        value_portfolio.name = year_str
        value_portfolio_df_list.append(value_portfolio)

        long_mom_weighted = mc_t.loc[long_mom.index] / mc_t.loc[long_mom.index].sum()
        short_mom_weighted = -mc_t.loc[short_mom.index] / mc_t.loc[short_mom.index].sum()
        mom_portfolio = pd.concat([long_mom_weighted, short_mom_weighted])
        mom_portfolio.name = year_str
        mom_portfolio_df_list.append(mom_portfolio)
    
    value_portfolio_df = pd.concat(value_portfolio_df_list, axis=1).fillna(0)
    mom_portfolio_df = pd.concat(mom_portfolio_df_list, axis=1).fillna(0)
    mixed_portfolio_df = value_portfolio_df.add(mom_portfolio_df, fill_value=0) / 2

    
    shifted_value_weights = value_portfolio_df.shift(1, axis=1)
    shifted_mom_weights = mom_portfolio_df.shift(1, axis=1)
    shifted_mixed_weights = mixed_portfolio_df.shift(1, axis=1)

    common_years = annual_return_df.columns.intersection(shifted_value_weights.columns).drop(str(start_year))
    
    aligned_returns = annual_return_df[common_years]
    aligned_value_weights = shifted_value_weights[common_years].reindex(aligned_returns.index).fillna(0)
    aligned_mom_weights = shifted_mom_weights[common_years].reindex(aligned_returns.index).fillna(0)
    aligned_mixed_weights = shifted_mixed_weights[common_years].reindex(aligned_returns.index).fillna(0)

    value_return_s = (aligned_value_weights * aligned_returns).sum(axis=0)
    mom_return_s = (aligned_mom_weights * aligned_returns).sum(axis=0)
    mixed_return_s = (aligned_mixed_weights * aligned_returns).sum(axis=0)

    aligned_rf_df = annual_rf_df.loc[value_return_s.index]
    all_returns_df = pd.concat([value_return_s, mom_return_s, mixed_return_s, aligned_rf_df], axis=1)
    all_returns_df.columns = ['Value Return', 'Momentum Return', '50/50 Portfolio Return', 'Risk-Free Rate']
    
    return all_returns_df

def prepare_data_for_mcmc(portfolio_df: pd.DataFrame):
    y = np.array(portfolio_df[['Value Return', 'Momentum Return']].values, dtype=np.float32).transpose() # (2, T)
    # X[:, t] = I_2 kronecker product [1, y[0, t-1], y[1, t-1]],
    # Construct X such that for each t, X[t] = I_2 ⊗ [1, y[0, t-1], y[1, t-1]]
    T = y.shape[1]
    X = np.zeros((T, 2, 6), dtype=np.float32) # (T, 2, 6)
    for t in range(T):
        if t == 0:
            prev_y = np.zeros(2, dtype=np.float32)
        else:
            prev_y = y[:, t-1]
        x_row = np.array([1, prev_y[0], prev_y[1]], dtype=np.float32)
        X[t] = np.kron(np.eye(2), x_row)
    return X, y

def run_mcmc_sampling(X: np.ndarray, y: np.ndarray):
    X = pt.tensor.as_tensor_variable(X)
    y = pt.tensor.as_tensor_variable(y)
    with pm.Model() as model:
        sigma_a_sq = pm.InverseGamma('sigma_a_sq', alpha=d_a0, beta=d_a1)
        sigma_h_sq = pm.InverseGamma('sigma_h_sq', alpha=d_h0, beta=d_h1)
        phi = pm.Gamma('phi', alpha=d, beta=d, shape=6)
        kappa_sq = pm.Gamma('kappa_sq', alpha=c_0, beta=c_1)
        theta = pm.InverseGamma('theta', alpha=0.5, beta=1/(2*phi * kappa_sq), shape=6)
        Q = pm.Deterministic('Q', pt.tensor.diag(theta))
        beta_0 = pm.MvNormal('beta_0', mu=pt.tensor.zeros(shape=6), cov=pt.tensor.eye(6) * 10)
        a_vm0 = pm.Normal('a_vm0', mu=0, sigma=10)
        log_h_0 = pm.Normal('log_h_0', mu=0, sigma=10, shape=2)

        # Random Walks
        betas = [beta_0]
        a_vms = [a_vm0]
        log_hs = [log_h_0]

        # State Transition Equation
        for t in range(1, T):
            beta_tm1 = betas[t-1]
            a_vm_tm1 = a_vms[t-1]
            log_h_tm1 = log_hs[t-1]
            beta_t = pm.MvNormal(f'beta_{t}', mu=beta_tm1, cov=Q, shape=6)
            a_vm_t = pm.Normal(f'a_vm_{t}', mu=a_vm_tm1, sigma=pt.tensor.sqrt(sigma_a_sq))
            log_h_t = pm.Normal(f'log_h_{t}', mu=log_h_tm1, sigma=pt.tensor.sqrt(sigma_h_sq), shape=2)

            A_t = pt.tensor.set_subtensor(pt.tensor.eye(2)[1, 0], a_vm_t)
            H_t = pt.tensor.diag(pt.tensor.exp(log_h_t))
            A_inv_t = pt.tensor.linalg.inv(A_t)
            Omega_t = A_inv_t @ H_t @ A_inv_t.T # shape (2, 2)

            pm.MvNormal(f'likelihood_{t}', mu=X[t] @ beta_t, cov=Omega_t, observed=y[:, t])
            betas.append(beta_t)
            a_vms.append(a_vm_t)
            log_hs.append(log_h_t)

    with model:
        trace = pm.sample(draws=SAMPLE_SIZE, tune=TUNE, chains=chains, target_accept=target_accept, return_inferencedata=True, random_seed=SEED, progressbar=True)
    
    az.to_netcdf(trace, f'{RESULT_PATH}/mcmc_trace.nc')
    return trace

def analyze_mcmc_trace(trace):
    """
    MCMC Trace 분석 함수
    """
    print("MCMC Trace Analysis:")
    print(az.summary(trace, round_to=2))
    
    # Plotting the trace
    az.plot_trace(trace)
    plt.tight_layout()
    plt.savefig(f'{RESULT_PATH}/mcmc_trace_plot.png', dpi=300)
    
    # Posterior predictive checks
    az.plot_ppc(trace)
    plt.tight_layout()
    plt.savefig(f'{RESULT_PATH}/mcmc_ppc_plot.png', dpi=300)

if __name__ == "__main__":
    port_return_df = compute_mom_and_value_return()
    print("=" * 60)
    print("Corrected Annual Portfolio Returns")
    print("=" * 60)
    print(port_return_df)
    print(f"\nCorrelation between Value and Momentum Returns: {port_return_df.corr().iloc[0, 1]:.4f}")
    print("=" * 60)
    
    # 누적 수익률 그래프
    cumulative_returns = (1 + port_return_df[['Value Return', 'Momentum Return', '50/50 Portfolio Return']]).cumprod()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=cumulative_returns-1, dashes=False, marker='o')
    plt.title('Cumulative Portfolio Returns')
    plt.xlabel('Year')
    plt.ylabel('Cumulative Return')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{RESULT_PATH}/value_mom_cumulative_returns.png', dpi=300)

    # 샤프 비율 바 차트
    sharpe_ratios = (port_return_df[['Value Return', 'Momentum Return', '50/50 Portfolio Return']].mean() - port_return_df['Risk-Free Rate'].mean()) / port_return_df[['Value Return', 'Momentum Return', '50/50 Portfolio Return']].std()
    print(f"Sharpe Ratios:\n{sharpe_ratios}")
    print("=" * 60)

    plt.figure(figsize=(8, 5))
    sharpe_ratios.plot(kind='bar', color=['blue', 'orange', 'green'])
    plt.title('Sharpe Ratios of Portfolios')
    plt.xlabel('Portfolio')
    plt.ylabel('Sharpe Ratio')
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{RESULT_PATH}/value_mom_sharpe_ratios.png', dpi=300)

    # MCMC Analysis
    print("=" * 60)
    print("Starting MCMC Analysis...")
    if LOAD and os.path.exists(f'{RESULT_PATH}/mcmc_trace.nc'):
        print("Loading existing MCMC trace from netCDF file...")
        trace = az.from_netcdf(f'{RESULT_PATH}/mcmc_trace.nc')
    else:
        print("netCDF File does not exist. Starting MCMC Sampling...")
        X, y = prepare_data_for_mcmc(port_return_df)
        trace = run_mcmc_sampling(X, y)
        print("MCMC Analysis Completed. Trace saved to netCDF file.")
    analyze_mcmc_trace(trace)


