#%% 
import pandas as pd
import numpy as np

#%%
merged_df = pd.read_csv("data /merged_df.csv", index_col=0)

# %% Preparing Dataset for the Simulation with relevant variables.

## Reorganizing the table into four aligned panels, to avoid misalignement  
# Identify column groups by suffix
debt_cols = [c for c in merged_df.columns if c.endswith(' Debt')]
ma_cols   = [c for c in merged_df.columns if c.endswith(' MA-5')]
pd_cols   = [c for c in merged_df.columns if c.endswith(' PDs')]
acc_cols  = [c for c in merged_df.columns if c.endswith(' Market Access')]

# Build cleaned panels (columns = country names without suffix)
def strip_block(cols, suffix): 
    return [c.replace(suffix,'').strip() for c in cols]

countries = sorted(
    set(strip_block(debt_cols,' Debt')) &
    set(strip_block(ma_cols,' MA-5')) &
    set(strip_block(acc_cols,' Market Access')) &
    set(strip_block(pd_cols,' PDs'))
)

Debt = merged_df[[f"{c} Debt" for c in countries]].astype(float)
Debt.columns = countries

MA = merged_df[[f"{c} MA-5" for c in countries]].astype(float)
MA.columns = countries

ACC = merged_df[[f"{c} Market Access" for c in countries]].astype(int)
ACC.columns = countries

PDs = merged_df[[f"{c} PDs" for c in countries]].astype(float)
PDs.columns = countries

# %% Builidng monthly weight function to later weight the portfolio accordingly - weight in function of GDP with a cap (k=0.25) on the participation of each country
def build_monthly_weights(gdp_row: pd.Series,
                          market_access_row: pd.Series,
                          outstanding_row: pd.Series,
                          k: float = 0.25,
                          eps: float = 1e-12) -> pd.Series:
    """
    One month:
      - exclude countries with market_access==0
      - share_i = MA_i / total over eligible
      - allocate amounts with S = 1.0
      - cap each alloc_i at k * outstanding_i and redistribute residual by room
      - return final SHARES (sum to 1)
    """
    gdp = gdp_row.fillna(0.0).astype(float)
    acc = market_access_row.fillna(0).astype(int)
    out = outstanding_row.fillna(0.0).astype(float)

    # eligible & renormalize MA weights
    eligible = (acc == 1)
    total = float(gdp[eligible].sum())
    if total <= eps:
        return pd.Series(0.0, index=gdp.index)

    w = pd.Series(0.0, index=gdp.index, dtype=float)
    w[eligible] = gdp[eligible] / total  # target shares

    # amounts world with S = 1.0
    S = 1.0
    cap_amt = k * out
    target_amt = w * S
    alloc = target_amt.clip(upper=cap_amt)  # initial clip

    # redistribute any residual to eligible countries with remaining room
    residual = S - float(alloc.sum())
    loops, max_loops = 0, 100
    while residual > eps and loops < max_loops:
        loops += 1
        room = (cap_amt - alloc).clip(lower=0.0)
        room[~eligible] = 0.0  
        total_room = float(room.sum())
        if total_room <= eps:
            break
        alloc += residual * (room / total_room)
        residual = S - float(alloc.sum())

    # final shares
    S_used = float(alloc.sum())
    final_shares = (alloc / S_used) if S_used > 0 else pd.Series(0.0, index=gdp.index)
    return final_shares


#%% Run it for every month

k = 0.25
final_weights = pd.DataFrame(index=MA.index, columns=countries, dtype=float)

for t in MA.index:
    final_weights.loc[t] = build_monthly_weights(
        gdp_row=MA.loc[t],
        market_access_row=ACC.loc[t],
        outstanding_row=Debt.loc[t],
        k=k
    )

#%%
# which countries hit the cap each month? 1=capped, 0=not
def capped_flags_for_month(w_shares: pd.Series, out_row: pd.Series, k=0.25, eps=1e-12) -> pd.Series:
    """
    A name is 'capped' if its final share equals (within tol) its capacity share:
      cap_share_i = (k*O_i) / sum_j (k*O_j) = O_i / sum_j O_j
    """
    O = out_row.fillna(0.0).astype(float)
    denom = float(O.sum())
    if denom <= eps:
        return pd.Series(0, index=w_shares.index)
    cap_share = O / denom
    tol = 1e-12
    return (w_shares >= cap_share - tol).astype(int)

capped_flags = pd.DataFrame(index=MA.index, columns=countries, dtype=int)
for t in MA.index:
    capped_flags.loc[t] = capped_flags_for_month(final_weights.loc[t], Debt.loc[t], k=k)

#%% Save results 
final_weights.to_csv("data /final_weights_k25.csv", float_format="%.6f")
capped_flags.to_csv("data /capped_flags.csv")

# %% Merging necessary data for the MC simulation

PDs_weights = pd.merge(PDs, final_weights, left_index=True, right_index=True, suffixes=('_PDs', '_Weights'))
PDs_weights.to_csv("data /PDs+Weights.csv")

# %%
