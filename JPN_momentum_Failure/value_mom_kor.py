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
import arviz as az

######################## DATA PREPARATION ########################
SEED = 42
np.random.seed(SEED) # for reproducibility
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

def compute_mom_and_value_return():
    """
    Parses the data from a file where each line contains two space-separated floats.    
    """
    # Annualized returns calculation
    annual_return_df_list = []
    for year in range(start_year, end_year + 1):
        temp_return_df = returns_df[returns_df.columns[returns_df.columns.str.startswith(str(year))]]
        annual_return_df = (1 + temp_return_df).prod(axis=1) - 1
        annual_return_df_list.append(annual_return_df)
    annual_return_df = pd.concat(annual_return_df_list, axis=1)
    annual_return_df.columns = [str(year) for year in range(start_year, end_year + 1)]

    # Annualized Mask Calculation: If there exists 1 in the year, then the mask is 1, else 0
    annual_mask_df_list = []
    for year in range(start_year, end_year + 1):
        temp_mask_df = mask_df[mask_df.columns[mask_df.columns.str.startswith(str(year))]]
        annual_mask_df = temp_mask_df.max(axis=1)
        annual_mask_df_list.append(annual_mask_df)
    annual_mask_df = pd.concat(annual_mask_df_list, axis=1)
    annual_mask_df.columns = [str(year) for year in range(start_year, end_year + 1)]
    annual_mask_df = annual_mask_df.fillna(0).astype(int)

    value_df = be_df / mc_df
    value_df = value_df.fillna(0)

    # Annualilzed Value Calculation: The last value of the year is used as the annual value
    annual_value_df_list = []
    for year in range(start_year, end_year + 1):
        temp_value_df = value_df[value_df.columns[value_df.columns.str.startswith(str(year))]]
        annual_value_df = temp_value_df.iloc[:, -1]
        annual_value_df_list.append(annual_value_df)
    annual_value_df = pd.concat(annual_value_df_list, axis=1)
    annual_value_df.columns = [str(year) for year in range(start_year, end_year + 1)]

    # Annualized Momentum Calculation: Average of the returns over the year, excluding the last month
    annual_mom_df_list = []
    for year in range(start_year, end_year + 1):
        temp_return_df = returns_df[returns_df.columns[returns_df.columns.str.startswith(str(year))]]
        temp_return_df = temp_return_df.drop(temp_return_df.columns[temp_return_df.columns.str.contains('-12-')], axis=1)
        annual_mom_df = (1 + temp_return_df).prod(axis=1) - 1
        annual_mom_df_list.append(annual_mom_df)
    annual_mom_df = pd.concat(annual_mom_df_list, axis=1)
    annual_mom_df.columns = [str(year) for year in range(start_year, end_year + 1)]

    # Annualized Market Capitalization Calculation: The last market cap of the year is used as the annual market cap
    annual_mc_df_list = []
    for year in range(start_year, end_year + 1):
        temp_mc_df = mc_df[mc_df.columns[mc_df.columns.str.startswith(str(year))]]
        annual_mc_df = temp_mc_df.iloc[:, -1]
        annual_mc_df_list.append(annual_mc_df)
    annual_mc_df = pd.concat(annual_mc_df_list, axis=1)
    annual_mc_df.columns = [str(year) for year in range(start_year, end_year + 1)]

    # Annualized Risk-Free Rate Calculation: Average of the risk-free rate over the year
    annual_rf_df_list = []
    for year in range(start_year, end_year + 1):
        temp_rf_df = rf_df[rf_df.columns[rf_df.columns.str.startswith(str(year))]]
        temp_rf_df = temp_rf_df.mean(axis=1)
        annual_rf_df_list.append(temp_rf_df)
    annual_rf_df = pd.concat(annual_rf_df_list, axis=1)
    annual_rf_df.columns = [str(year) for year in range(start_year, end_year + 1)]
    annual_rf_df = annual_rf_df.transpose().iloc[:, 0].to_frame(name='Risk-Free Rate')


    # Value & Momentum Portfolio Construction: Long Top 33%, Short Bottom 33% with dollar neutralization, weighted by market cap
    value_portfolio_df_list = []
    mom_portfolio_df_list = []
    for year in range(start_year, end_year + 1):
        year = str(year)
        members = annual_mask_df[year][annual_mask_df[year]==1].index
        if members.empty:
            continue

        # Value and Momentum Quantile Break points
        value = annual_value_df.loc[members, year].dropna()
        mom = annual_mom_df.loc[members, year].dropna()
        value_brp = value.quantile([0.33, 0.67]).values
        mom_brp = mom.quantile([0.33, 0.67]).values

        # Long and Short Portfolios Members
        long_value = value[value >= value_brp[1]]
        short_value = value[value <= value_brp[0]]
        long_mom = mom[mom >= mom_brp[1]]
        short_mom = mom[mom <= mom_brp[0]]

        # Value Portfolio
        long_value_weighted = annual_mc_df.loc[long_value.index, year]
        short_value_weighted = annual_mc_df.loc[short_value.index, year]
        long_value_weighted = long_value_weighted / long_value_weighted.sum()
        short_value_weighted = -short_value_weighted / short_value_weighted.sum()
        value_portfolio = pd.concat([long_value_weighted, short_value_weighted], axis=0)
        value_portfolio_df_list.append(value_portfolio)

        # Momentum Portfolio
        long_mom_weighted = annual_mc_df.loc[long_mom.index, year]
        short_mom_weighted = annual_mc_df.loc[short_mom.index, year]
        long_mom_weighted = long_mom_weighted / long_mom_weighted.sum()
        short_mom_weighted = -short_mom_weighted / short_mom_weighted.sum()
        mom_portfolio = pd.concat([long_mom_weighted, short_mom_weighted], axis=0)
        mom_portfolio_df_list.append(mom_portfolio)
    
    # Concatenating all portfolios
    value_portfolio_df = pd.concat(value_portfolio_df_list, axis=1)
    value_portfolio_df.columns = [str(year) for year in range(start_year, end_year + 1)]
    value_portfolio_df = value_portfolio_df.fillna(0)
    mom_portfolio_df = pd.concat(mom_portfolio_df_list, axis=1)
    mom_portfolio_df.columns = [str(year) for year in range(start_year, end_year + 1)]
    mom_portfolio_df = mom_portfolio_df.fillna(0)
    # Mixed Portfolio: Value + Momentum, members may not be the same
    mixed_portfolio_df = value_portfolio_df.add(mom_portfolio_df, fill_value=0)

    # Value and Momentum Returns Calculation
    value_return_df = (value_portfolio_df * annual_return_df.loc[value_portfolio_df.index]).sum(axis=0)
    mom_return_df = (mom_portfolio_df * annual_return_df.loc[mom_portfolio_df.index]).sum(axis=0)
    mixed_portfolio_df = (mixed_portfolio_df * annual_return_df.loc[mixed_portfolio_df.index]).sum(axis=0)

    # Concatenating all returns
    all_returns_df = pd.concat([value_return_df, mom_return_df, mixed_portfolio_df, annual_rf_df], axis=1)
    all_returns_df.columns = ['Value Return', 'Momentum Return', '50/50 Portfolio Return', 'Risk-Free Rate']
    return all_returns_df

def temp():
    """
    Temporary function to avoid errors in the code.
    """
    pass



if __name__ == "__main__":
    port_return_df = compute_mom_and_value_return()
    print("Value and Momentum Returns Computation Completed.")
    print("=" * 50)
    print(port_return_df)
    print(f"Correlation between Value and Momentum Returns: {port_return_df.corr().iloc[0, 1]:.4f}")
    print("=" * 50)
    
    # plotting the returns
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=port_return_df, dashes=False)
    plt.title('Value and Momentum Returns Over Time')
    plt.xlabel('Year')
    plt.ylabel('Annualized Return')
    plt.legend(title='Portfolio', labels=['Value Return', 'Momentum Return'])
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{RESULT_PATH}/value_mom_returns.png', dpi=300)

    # bar chart of sharpe ratios=(mean return - risk-free rate) / std deviation of returns
    sharpe_ratios = (port_return_df.mean() - port_return_df['Risk-Free Rate'].mean()) / port_return_df.std()
    print(sharpe_ratios)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=sharpe_ratios.index, y=sharpe_ratios.values)
    plt.title('Sharpe Ratios of Value and Momentum Portfolios')
    plt.xlabel('Portfolio')

