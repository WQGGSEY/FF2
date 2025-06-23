import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import os
import pymc as pm
import pytensor as pt
import arviz as az
from tqdm import tqdm
import xarray as xr

######################## DATA PREPARATION ########################
# (기존 코드와 동일)
file_path='simulation_data'
start_date = '2015-01-02'
end_date = '2024-12-30'
start_year = int(start_date.split('-')[0])
end_year = int(end_date.split('-')[0])
# ... (데이터 로딩 부분은 기존과 동일)
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
with open(f"{file_path}/rf_bond.pkl", 'rb') as f:
    rf_df:pd.DataFrame = pickle.load(f).ffill(axis=1) * 0.01
    rf_df = rf_df.loc[:, start_date:end_date]
#################################################################
RESULT_PATH = 'JPN_momentum_Failure/results'
if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)
#################################################################

################ HyperParameter For MCMC (수정됨) ################
SEED = 42
LOAD = True  # 재실행을 위해 False로 변경 권장
SAMPLE_SIZE = 2000
TUNE = 1000
CHAINS = 4
TARGET_ACCEPT = 0.99

# 사전분포 스케일 파라미터 (새로 추가/수정)
BETA_SIGMA_A = 0.05
BETA_SIGMA_H = 0.05
BETA_PHI_KAPPA = 0.05
INITIAL_STATE_SIGMA = .5 # 초기 상태 사전 분포 표준편차
#################################################################

# compute_mom_and_value_return() 함수는 기존과 동일
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

    X_pt = pt.tensor.as_tensor_variable(X, name='X')
    y_pt = pt.tensor.as_tensor_variable(y, name='y')
    T = y.shape[1]
    
    with pm.Model() as robust_model:
        # 분산 파라미터에 HalfCauchy 사용 (0 근처에서 더 안정적)
        sigma_a = pm.HalfCauchy('sigma_a', beta=BETA_SIGMA_A)
        sigma_h = pm.HalfCauchy('sigma_h', beta=BETA_SIGMA_H, shape=2)

        # theta에 대한 계층적 축소(Hierarchical Shrinkage) 적용
        # theta_i가 독립적이라고 가정하는 대신, 공통 분포에서 왔다고 가정
        phi = pm.HalfNormal('phi', sigma=BETA_PHI_KAPPA) # Global shrinkage parameter
        kappa_sq = pm.HalfNormal('kappa_sq', sigma=BETA_PHI_KAPPA) # Global shrinkage parameter
        
        # theta의 각 원소는 공통 분포를 따름 (Local shrinkage)
        theta_raw = pm.HalfNormal('theta_raw', sigma=1.0, shape=6)
        theta = pm.Deterministic('theta', theta_raw * pt.tensor.sqrt(1 / (2 * phi * kappa_sq)))
        
        # Q의 Cholesky 분해
        Q_sqrt = pt.tensor.diag(pt.tensor.sqrt(theta)) # theta는 분산이므로 sqrt를 취해야 표준편차

        # 초기 상태에 대한 사전 분포의 분산을 줄여 탐색 공간을 제한
        beta_0 = pm.MvNormal('beta_0', mu=pt.tensor.zeros(6), cov=pt.tensor.eye(6) * INITIAL_STATE_SIGMA)
        a_vm0 = pm.Normal('a_vm0', mu=0, sigma=INITIAL_STATE_SIGMA)
        log_h_0 = pm.Normal('log_h_0', mu=0, sigma=INITIAL_STATE_SIGMA, shape=2)

        betas = [beta_0]
        a_vms = [a_vm0]
        log_hs = [log_h_0]
        
        for t in range(T):
            beta_tm1 = betas[t]
            a_vm_tm1 = a_vms[t]
            log_h_tm1 = log_hs[t]
            
            # 비중심화된 방식의 혁신(innovation) 샘플링
            beta_offset = pm.MvNormal(f'beta_offset_{t+1}', mu=pt.tensor.zeros(6), cov=pt.tensor.eye(6))
            a_vm_offset = pm.Normal(f'a_vm_offset_{t+1}', mu=0, sigma=1)
            log_h_offset = pm.Normal(f'log_h_offset_{t+1}', mu=0, sigma=1, shape=2)

            # 결정론적으로 다음 상태 계산
            beta_t = beta_tm1 + Q_sqrt @ beta_offset
            a_vm_t = a_vm_tm1 + sigma_a * a_vm_offset
            log_h_t = log_h_tm1 + sigma_h * log_h_offset

            # Likelihood 계산
            A_t = pt.tensor.set_subtensor(pt.tensor.eye(2)[1, 0], a_vm_t)
            H_t_diag = pt.tensor.exp(log_h_t)
            
            # Likelihood의 공분산 행렬 계산
            A_inv_t = pt.tensor.linalg.inv(A_t)
            A_inv_H = A_inv_t * H_t_diag[None, :] 
            Omega_t = A_inv_H @ A_inv_t.T

            pm.MvNormal(f'likelihood_{t}', mu=X_pt[t] @ beta_t, cov=Omega_t, observed=y_pt[:, t])
            
            betas.append(beta_t)
            a_vms.append(a_vm_t)
            log_hs.append(log_h_t)

        all_betas = pm.Deterministic('all_betas', pt.tensor.stack(betas[1:]))
        all_log_hs = pm.Deterministic('all_log_hs', pt.tensor.stack(log_hs[1:]))
        all_a_vms = pm.Deterministic('all_a_vms', pt.tensor.stack(a_vms[1:]))

        # --- 샘플링 (수정된 하이퍼파라미터 적용) ---
        trace = pm.sample(
            draws=SAMPLE_SIZE, 
            tune=TUNE, 
            chains=CHAINS, 
            target_accept=TARGET_ACCEPT, 
            random_seed=SEED,
            progressbar=True
        )
    
    # --- 사후 예측 샘플링 (PPC Plot을 위해 추가) ---
    with robust_model:
        pm.sample_posterior_predictive(trace, extend_inferencedata=True)

    az.to_netcdf(trace, f'{RESULT_PATH}/mcmc_trace_{SEED}_{SAMPLE_SIZE}_{TUNE}_{CHAINS}.nc')
    return trace


def analyze_mcmc_trace(trace, X, y):

    print("=" * 60)
    print("MCMC Analysis Summary:")
    print("=" * 60)
    summary = az.summary(trace)
    print(summary)
    
    if (summary['r_hat'] > 1.01).any():
        print("\n[Warning] Some parameters have R-hat values exceeding 1.01. There may be convergence issues.")
        print(summary[summary['r_hat'] > 1.01])
    else:
        print("\n[Success] All parameters have R-hat values below 1.01, indicating good convergence.")

    min_ess = CHAINS * 100
    if (summary['ess_bulk'] < min_ess).any() or (summary['ess_tail'] < min_ess).any():
        print(f"\n[Warning] Some parameters have effective sample sizes (ESS) below {min_ess}.")
        print(summary[(summary['ess_bulk'] < min_ess) | (summary['ess_tail'] < min_ess)])
    else:
        print(f"\n[Success] All parameters have effective sample sizes (ESS) above {min_ess}.")

    print("Saving Trace Plot to file...")
    # 진단이 필요한 핵심 파라미터 위주로 trace plot 확인
    az.plot_trace(trace, var_names=['sigma_a', 'sigma_h', 'phi', 'kappa_sq', 'beta_0', 'a_vm0'])
    plt.tight_layout()
    plt.savefig(f'{RESULT_PATH}/mcmc_trace_plot.png', dpi=300)
    plt.close()

    print("Saving Posterior Predictive Check (PPC) plot to file...")
    try:
        # pm.sample_posterior_predictive를 통해 생성된 데이터를 사용
        az.plot_ppc(trace, num_pp_samples=100, group="posterior_predictive")
        plt.tight_layout()
        plt.savefig(f'{RESULT_PATH}/mcmc_ppc_plot.png', dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error generating PPC plot: {e}")
        print("Generating PPC plot manually using existing trace and data...")
        # 수동으로 PPC 샘플을 생성하는 함수 호출
        generate_ppc_manually(f'{RESULT_PATH}/mcmc_trace_{SEED}_{SAMPLE_SIZE}_{TUNE}_{CHAINS}.nc', X, y)

def generate_ppc_manually(trace_path: str, X: np.ndarray, y: np.ndarray):

    print(f"Loading existing trace from: {trace_path}")
    if not os.path.exists(trace_path):
        print(f"Error: Trace file not found.")
        return
    trace = az.from_netcdf(trace_path)
    
    print("Extracting posterior samples...")
    posterior = trace.posterior
    all_betas = posterior["all_betas"].values
    all_a_vms = posterior["all_a_vms"].values
    all_log_hs = posterior["all_log_hs"].values
    
    n_chains, n_draws, n_timesteps, _ = all_betas.shape
    
    print("Manually generating posterior predictive samples...")
    ppc_samples = np.zeros((n_chains, n_draws, n_timesteps, y.shape[0]))
    
    for chain in tqdm(range(n_chains), desc="Chains"):
        for draw in range(n_draws):
            for t in range(n_timesteps):
                beta_t = all_betas[chain, draw, t, :]
                a_vm_t = all_a_vms[chain, draw, t]
                log_h_t = all_log_hs[chain, draw, t, :]
                x_t = X[t, :, :]
                
                mu_t = x_t @ beta_t
                A_t = np.eye(2)
                A_t[1, 0] = a_vm_t
                H_t_diag = np.exp(log_h_t)
                
                try:
                    A_inv_t = np.linalg.inv(A_t)
                    A_inv_H = A_inv_t * H_t_diag[None, :] 
                    Omega_t = A_inv_H @ A_inv_t.T
                    sample = np.random.multivariate_normal(mean=mu_t, cov=Omega_t, size=1)
                    ppc_samples[chain, draw, t, :] = sample
                except np.linalg.LinAlgError:
                    ppc_samples[chain, draw, t, :] = np.nan

    print("Replacing predictive groups in the trace object...")
    
    # ArviZ 데이터셋 생성
    ppc_dataset = xr.Dataset(
        {"likelihood": (("chain", "draw", "time", "y_dim"), ppc_samples)},
        coords={"chain": np.arange(n_chains), "draw": np.arange(n_draws), 
                "time": np.arange(n_timesteps), "y_dim": np.arange(y.shape[0])}
    )
    obs_dataset = xr.Dataset(
        {"likelihood": (("time", "y_dim"), y.T)},
        coords={"time": np.arange(n_timesteps), "y_dim": np.arange(y.shape[0])}
    )

    trace.posterior_predictive = ppc_dataset
    trace.observed_data = obs_dataset
    
    print("Generating and saving the corrected PPC plot...")
    az.plot_ppc(trace, num_pp_samples=100)
    plt.tight_layout()
    plt.savefig(f'{RESULT_PATH}/mcmc_ppc_plot_repaired.png', dpi=300)
    plt.close()
    print("PPC plot successfully generated and saved as 'mcmc_ppc_plot_repaired.png'.")

def calculate_optimal_portfolio(trace, X, y, annual_returns_df):
    """
    MCMC trace를 기반으로 연도별 최적 포트폴리오 가중치와 샤프 비율을 계산합니다.

    Args:
        trace (az.InferenceData): MCMC 샘플링 결과
        X (np.ndarray): 설명변수 행렬
        y (np.ndarray): 실제 수익률 데이터 (2, T)
        annual_returns_df (pd.DataFrame): 무위험 이자율이 포함된 연간 수익률 데이터프레임

    Returns:
        pd.DataFrame: 연도별 최적 가중치
        float: 최적 포트폴리오의 샤프 비율
    """
    print("\n" + "="*60)
    print("Calculating Optimal Portfolio Weights from MCMC Trace...")
    print("="*60)

    # 1. Trace에서 사후 샘플 추출
    posterior = trace.posterior
    all_betas = posterior["all_betas"].values
    all_a_vms = posterior["all_a_vms"].values
    all_log_hs = posterior["all_log_hs"].values
    
    n_chains, n_draws, n_timesteps, _ = all_betas.shape
    
    # 2. 각 샘플/시간별 최적 가중치를 저장할 배열 초기화
    #    (Value 포트폴리오의 가중치만 저장)
    optimal_weights_dist = np.zeros((n_chains, n_draws, n_timesteps))
    
    # 3. 무위험 이자율 추출
    risk_free_rates = annual_returns_df['Risk-Free Rate'].values

    # 4. 모든 MCMC 샘플에 대해 최적 가중치 계산
    for chain in tqdm(range(n_chains), desc="Optimizing Weights (Chains)"):
        for draw in range(n_draws):
            for t in range(n_timesteps):
                # 특정 샘플의 파라미터 값 추출
                beta_t = all_betas[chain, draw, t, :]
                a_vm_t = all_a_vms[chain, draw, t]
                log_h_t = all_log_hs[chain, draw, t, :]
                x_t = X[t, :, :]
                
                # 해당 시점의 예상 수익률(mu)과 공분산(Omega) 재구성
                mu_t = x_t @ beta_t
                A_t = np.eye(2)
                A_t[1, 0] = a_vm_t
                H_t_diag = np.exp(log_h_t)
                
                try:
                    A_inv_t = np.linalg.inv(A_t)
                    Omega_t = (A_inv_t * H_t_diag[None, :]) @ A_inv_t.T
                    
                    # 샤프 비율 최대화 가중치 계산 (수학적 해)
                    rf_t = risk_free_rates[t]
                    inv_Omega_t = np.linalg.inv(Omega_t)
                    excess_returns = mu_t - rf_t
                    
                    # 정규화되지 않은 가중치
                    unnormalized_weights = inv_Omega_t @ excess_returns
                    
                    # 합이 1이 되도록 정규화
                    total_weight = np.sum(unnormalized_weights)
                    if total_weight != 0:
                        normalized_weights = unnormalized_weights / total_weight
                    else:
                        normalized_weights = np.array([0.5, 0.5]) # 합이 0일 경우 등분

                    # Value 포트폴리오의 가중치 저장
                    optimal_weights_dist[chain, draw, t] = normalized_weights[0]
                    
                except np.linalg.LinAlgError:
                    # 행렬 계산 오류 시, 해당 샘플은 NaN으로 처리
                    optimal_weights_dist[chain, draw, t] = np.nan

    # 5. 각 연도별 최적 가중치의 평균 계산
    #    NaN 값을 무시하고 평균을 계산합니다.
    mean_optimal_weights = np.nanmean(optimal_weights_dist, axis=(0, 1))
    
    # 6. 결과를 보기 좋은 데이터프레임으로 정리
    optimal_weights_df = pd.DataFrame({
        'Year': annual_returns_df.index,
        'Value_Weight': mean_optimal_weights,
        'Momentum_Weight': 1 - mean_optimal_weights
    }).set_index('Year')

    # 7. 계산된 최적 가중치를 사용하여 포트폴리오 수익률 시계열 생성
    value_returns = y[0, :]
    momentum_returns = y[1, :]
    
    optimal_portfolio_returns = \
        mean_optimal_weights * value_returns + (1 - mean_optimal_weights) * momentum_returns
        
    # 8. 최종 포트폴리오의 샤프 비율 계산
    optimal_portfolio_excess_returns = optimal_portfolio_returns - risk_free_rates
    mean_excess_return = np.mean(optimal_portfolio_excess_returns)
    std_dev_return = np.std(optimal_portfolio_excess_returns)
    
    optimal_sharpe_ratio = mean_excess_return / std_dev_return if std_dev_return != 0 else 0

    return optimal_weights_df, optimal_sharpe_ratio


if __name__ == "__main__":
    port_return_df = compute_mom_and_value_return()
    print("=" * 60)
    print("Annual Portfolio Returns")
    print("=" * 60)
    print(port_return_df)
    print(f"\nCorrelation between Value and Momentum Returns: {port_return_df.corr().iloc[0, 1]:.4f}")
    print("=" * 60)
    
    cumulative_returns = (1 + port_return_df[['Value Return', 'Momentum Return', '50/50 Portfolio Return']]).cumprod()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=cumulative_returns-1, dashes=False, marker='o')
    plt.title('Cumulative Portfolio Returns')
    plt.xlabel('Year')
    plt.ylabel('Cumulative Return')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{RESULT_PATH}/value_mom_cumulative_returns.png', dpi=300)

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
    print("Starting MCMC Analysis...")
    X, y = prepare_data_for_mcmc(port_return_df)
    if LOAD and os.path.exists(f'{RESULT_PATH}/mcmc_trace_{SEED}_{SAMPLE_SIZE}_{TUNE}_{CHAINS}.nc'):
        print("Loading existing MCMC trace from netCDF file...")
        trace = az.from_netcdf(f'{RESULT_PATH}/mcmc_trace_{SEED}_{SAMPLE_SIZE}_{TUNE}_{CHAINS}.nc')
    else:
        print("netCDF File does not exist. Starting MCMC Sampling...")
        trace = run_mcmc_sampling(X, y)
        print("MCMC Analysis Completed. Trace saved to netCDF file.")
    
    analyze_mcmc_trace(trace, X, y)

    mcmc_years = port_return_df.index[-y.shape[1]:]
    mcmc_annual_returns_df = port_return_df.loc[mcmc_years]
    
    optimal_weights, optimal_sharpe = calculate_optimal_portfolio(trace, X, y, mcmc_annual_returns_df)

    print("\n" + "="*60)
    print("Annual Optimal Portfolio Weights (Mean of Posterior)")
    print("="*60)
    print(optimal_weights.to_string(float_format="%.4f"))

    print("\n" + "="*60)
    print(f"Ex-Post Sharpe Ratio of Optimal Portfolio: {optimal_sharpe:.4f}")
    print("="*60)

    # 최적 가중치 시계열 시각화
    plt.figure(figsize=(12, 6))
    optimal_weights.plot(kind='bar', stacked=True, ax=plt.gca())
    plt.title('Optimal Portfolio Weights per Year')
    plt.ylabel('Weight')
    plt.xlabel('Year')
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    plt.legend(title='Portfolio')
    plt.tight_layout()
    plt.savefig(f'{RESULT_PATH}/optimal_weights_plot.png', dpi=300)
    plt.close()
    print("\nOptimal weights plot saved to 'optimal_weights_plot.png'")