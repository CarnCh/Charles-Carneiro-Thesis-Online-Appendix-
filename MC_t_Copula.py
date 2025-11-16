#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as student_t, rankdata
from scipy.special import gammaln

# 
path = "data /PDs+Weights.csv"  # note: the folder name contains a space after 'data'
df = pd.read_csv(path, index_col=0)
df.index = pd.to_datetime(df.index, errors="coerce").to_period("M").to_timestamp("M")
df = df.sort_index()

pd_cols = [c for c in df.columns if c.endswith("_PDs")]
w_cols  = [c for c in df.columns if c.endswith("_Weights")]
countries = sorted({c.replace("_PDs","") for c in pd_cols} & {c.replace("_Weights","") for c in w_cols})

PDs = df[[f"{c}_PDs" for c in countries]].astype(float).copy(); PDs.columns = countries
final_weights = df[[f"{c}_Weights" for c in countries]].astype(float).copy(); final_weights.columns = countries

# Model settings 
probabilities = np.array([0.70, 0.25, 0.05])              # [good, mild, severe]
base_LGD = 0.60
LGD_scalers = np.array([0.75, 1.0, 1.25])
hurdle = 0.005
sub_grid = np.round(np.linspace(0.00, 0.90, 91), 3)
n = 500_000
rng = np.random.default_rng(7)
Germany = "Germany"
sub_fixed = 0.30

# Student-t copula params
nu_tcopula = 15.0         # degrees of freedom (grid-selected per window below)
win_months = 60          # rolling window length in months

# Fixed-lambda scenario for robustness (set to None to disable)
fixed_lambda_scenario = 0.50   # 0.50 = moderate shrinkage; None = no scenario run

#%% Helpers
# function ensures a matrix is positive semi-definite (PSD)
# A correlation matrix must be PSD to be valid (otherwise Cholesky decomposition or multivariate sampling can break down)
def _nearest_psd(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    A = (A + A.T) / 2.0
    vals, vecs = np.linalg.eigh(A) # eigen-decomposition
    vals = np.clip(vals, eps, None) # clip negative eigenvalues to a small positive value, ensuring no negative values
    Apsd = (vecs * vals) @ vecs.T # reconstruct matrix
    D = np.diag(1.0 / np.sqrt(np.clip(np.diag(Apsd), 1e-16, None)))
    C = D @ Apsd @ D
    return (C + C.T) / 2.0

# add EL_senior at fixed s-levels into the row dict 
def add_el_checkpoints(row_dict: dict, EL_senior_norm_curve: dict, levels=(0,10,20,30,40,50,60,70)):
    """
    Store **normalized-to-tranche** senior ELs at fixed s-levels.
    levels are in percentage points (ints). Names: EL_sen_<pp> (e.g., EL_sen_30).
    Assumes your sub_grid contains these levels (step=0.01, rounded to 3 d.p.).
    """
    for pp in levels:
        s = round(pp / 100.0, 3)        # exact key as in your sub_grid (3 d.p.)
        key = f"EL_sen_{pp:02d}"
        row_dict[key] = EL_senior_norm_curve.get(s, float('nan'))
    return row_dict


# Diagnostics collector container
tcopula_diag = []   # will store per-month, per-ν rows
tcopula_diag_fixed = [] 
def _mean_offdiag(R: np.ndarray) -> float:
    d = R.shape[0]
    if d < 2: return 0.0
    s = (R.sum() - np.trace(R))
    return float(s / (d * (d - 1)))

#%% t-copula pseudo-likelihood
# Maximising the log-likelihood over R (correlation) and v (degrees of freedom)
def t_copula_loglik_from_z(z: np.ndarray, R: np.ndarray, nu: float) -> float:
    T, d = z.shape # T = n_samples, d = n_dimensions (sovereigns)
    L = np.linalg.cholesky(R) # Cholesky decomposition
    logdetR = 2.0 * np.sum(np.log(np.diag(L)))
    Rinv = np.linalg.inv(R)

    const_d = gammaln((nu + d) / 2.0) - gammaln(nu / 2.0) - 0.5 * d * np.log(nu * np.pi) # Normalizing constants for multivariate and univariate t
    const_1 = gammaln((nu + 1) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(nu * np.pi)

    Q = np.einsum('ti,ij,tj->t', z, Rinv, z) # Quadratic form (Mahalanobis distance), Measures how “far” each point is from center under correlation R

    ll_mv = T * (const_d - 0.5 * logdetR) - ((nu + d) / 2.0) * np.sum(np.log1p(Q / nu))
    ll_uni = T * z.shape[1] * const_1 - ((nu + 1) / 2.0) * np.sum(np.log1p((z ** 2) / nu))
    return float(ll_mv - ll_uni) # Return log-likelihood of the t-copula

# Helper: maximize t-copula pseudo-likelihood over lambda in [0,1]
# Correlations can be noisy and this function finds the optimal shrinkage parameter lambda
def _maximize_lambda_likelihood(z: np.ndarray, S: np.ndarray, nu: float,
                                tol: float = 1e-4, max_iter: int = 60) -> tuple[float, float]:
    """
    Golden-section search for lambda in [0,1] maximizing the t-copula pseudo-likelihood.
    Returns (lambda_star, ll_star).
    """
    I = np.eye(S.shape[0])

    def f(lam: float) -> float:
        lam = float(np.clip(lam, 0.0, 1.0))
        R = lam * I + (1.0 - lam) * S
        # R should be SPD
        try:
            return t_copula_loglik_from_z(z, R, nu)
        except np.linalg.LinAlgError:
            R_psd = _nearest_psd(R, eps=1e-10)
            return t_copula_loglik_from_z(z, R_psd, nu)

    # Evaluate endpoints too (maximum can be at boundary)
    f0 = f(0.0)
    f1 = f(1.0)

    # Golden-section search on (0,1)
    a, b = 0.0, 1.0
    invphi = (np.sqrt(5) - 1) / 2.0
    invphi2 = (3 - np.sqrt(5)) / 2.0
    x1 = a + invphi2 * (b - a)
    x2 = a + invphi * (b - a)
    f1x = f(x1)
    f2x = f(x2)

    it = 0
    while (b - a) > tol and it < max_iter:
        if f1x < f2x:  # keep right subinterval for maximization
            a = x1
            x1 = x2
            f1x = f2x
            x2 = a + invphi * (b - a)
            f2x = f(x2)
        else:
            b = x2
            x2 = x1
            f2x = f1x
            x1 = a + invphi2 * (b - a)
            f1x = f(x1)
        it += 1

    lam_star = x1 if f1x >= f2x else x2
    ll_star = max(f1x, f2x, f0, f1)
    # If an endpoint beats interior points, use it
    if f0 >= ll_star and f0 >= f1:
        lam_star, ll_star = 0.0, f0
    elif f1 >= ll_star and f1 >= f0:
        lam_star, ll_star = 1.0, f1

    return float(np.clip(lam_star, 0.0, 1.0)), float(ll_star)

#%% Pseudo-observations with gentle de-tying noise
# To focus on dependence, I convert raw data into pseudo-observations (uniforms in (0,1]) 
def window_pseudo_obs(hist_df: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """
    hist_df: T x d window (numeric), already ffilled/bfilled.
    Returns U (T x d) in (0,1], via columnwise ranks, with tiny random jitter to break ties.
    """
    X = hist_df.select_dtypes(include="number").to_numpy(float)
    T, d = X.shape
    U = np.empty_like(X, dtype=float)
    for j in range(d):
        r = rankdata(X[:, j], method="average")
        U[:, j] = r / (T + 1.0)
    # tiny jitter to avoid exact ties in PIT -> t-ppf; preserves ranks
    rng_local = np.random.default_rng(seed)
    eps = 1e-6 / (T + 2.0)
    U += (rng_local.random(U.shape) - 0.5) * eps
    U = np.clip(U, 1e-12, 1 - 1e-12)
    return U

#%% Estimate correlation and nu (grid) per rolling window; option to fix lambda
# Core function that estimates correlation matrix, tail heaviness (degrees of freedom), and the shrinkage parameter
def estimate_corr_and_nu_grid(PDs_full: pd.DataFrame,
                              t_anchor: pd.Timestamp,
                              active_cols: list[str],
                              window_months: int = 60,
                              nu_grid=( 4, 5, 7, 10,15),
                              fixed_lambda: float | None = None,
                              diag_store: list | None = None
                              ) -> tuple[np.ndarray, float, float, list[str]]:
    """
    Returns (R_hat, nu_hat, lambda_used, used_cols) for the window ending at t_anchor.
    If fixed_lambda is not None, uses R = fixed_lambda*I + (1-fixed_lambda)*S.
    λ is chosen by maximizing the t-copula pseudo-likelihood when fixed_lambda is None.
    ν is selected by an AIC comparison across the provided grid (k=1 for ν; λ is profiled).
    """
    start = (t_anchor.to_period("M") - window_months + 1).to_timestamp("M")
    hist_all = PDs_full.loc[(PDs_full.index >= start) & (PDs_full.index <= t_anchor), active_cols].sort_index()
    hist_all = hist_all.ffill().bfill()

    # keep only columns that have at least one non-NaN in the window
    valid_mask = hist_all.notna().any(axis=0)
    used_cols = [c for c in hist_all.columns if valid_mask[c]]
    hist = hist_all[used_cols].copy()

    T = hist.shape[0]
    d = hist.shape[1]
    if (d < 2) or (T < max(12, d + 2)):
        # fallback: identity correlation of size d (return a 4-tuple consistently)
        return np.eye(max(d, 1)), 15.0, 0, used_cols

    U = window_pseudo_obs(hist, seed=0)

    best = None
    for nu in nu_grid:
        z = student_t.ppf(U, df=nu)
        z -= z.mean(axis=0, keepdims=True)
        std = z.std(axis=0, ddof=0)
        z  /= np.clip(std, 1e-12, None)

        S = (z.T @ z) / z.shape[0]

        if fixed_lambda is not None:
            lam_hat = float(np.clip(fixed_lambda, 0.0, 1.0))
            R = lam_hat * np.eye(S.shape[0]) + (1.0 - lam_hat) * S
            try: np.linalg.cholesky(R)
            except np.linalg.LinAlgError: R = _nearest_psd(R, eps=1e-10)
            ll = t_copula_loglik_from_z(z, R, nu)
        else:
            lam_hat, ll = _maximize_lambda_likelihood(z, S, nu)
            R = lam_hat * np.eye(S.shape[0]) + (1.0 - lam_hat) * S
            try: np.linalg.cholesky(R)
            except np.linalg.LinAlgError: R = _nearest_psd(R, eps=1e-10)

        aic = 2 * 1 - 2 * ll  # k=1 (only ν counted; λ is profiled)

        # log diagnostics (one row per ν candidate) 
        if diag_store is not None:
            diag_store.append({
                "Month": t_anchor,
                "nu": float(nu),
                "lambda_hat": float(lam_hat),
                "loglik": float(ll),
                "AIC": float(aic),
                "T_window": int(U.shape[0]),
                "dim": int(S.shape[0]),
                "mean_offdiag": _mean_offdiag(R)
            })

        if (best is None) or (aic < best['aic']):
            best = {'R': R, 'nu': float(nu), 'lam': float(lam_hat), 'aic': float(aic)}
    return best['R'], best['nu'], best['lam'], used_cols


#%% t-copula sampler
# This function simulates dependent uniforms from a multivariate t-copula, enforcing correlation and heavy tails
def sample_t_copula_uniforms(n_paths: int, corr: np.ndarray, nu: float, rng: np.random.Generator) -> np.ndarray:
    d = corr.shape[0]
    try:
        L = np.linalg.cholesky(corr) # Cholesky decomposition
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(_nearest_psd(corr, eps=1e-10))
    Z = rng.standard_normal((n_paths, d)) @ L.T # simulate n paths and correlate
    S = rng.chisquare(df=nu, size=n_paths) / nu # add fat tail with chi-square scaling
    Tm = Z / np.sqrt(S)[:, None]     # multivariate student-t
    U = student_t.cdf(Tm, df=nu)     # PIT to uniforms
    return U

#%% State-splitting & LGD
# Split baseline PDs into state dependent PDs
def split_pd_states(pd5: np.ndarray, probabilities: np.ndarray):
    scale_input = np.asarray(pd5, dtype=float)
    factor = 100.0 if np.nanmax(scale_input) <= 1.0 else 1.0
    x = np.clip(scale_input * factor, 1e-12, None)
    ln_x = np.log(x)
    sev_good = np.ones_like(ln_x)
    sev_mild = 7.9265 * ln_x + 25.791
    sev_severe = 11.419 * ln_x + 59.964
    Sev = np.vstack([sev_good, sev_mild, sev_severe])
    scale = (probabilities.reshape(-1, 1) * Sev).sum(axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    PDs_SxN = (Sev / scale) * scale_input.reshape(1, -1)
    return np.clip(PDs_SxN, 0.0, 1.0)

# Construct a matrix of state dependent LGds 
def lgd_states_matrix(n_countries: int, base_LGD: float, LGD_scalers: np.ndarray):
    return (base_LGD * LGD_scalers)[:, None] * np.ones((1, n_countries))

# Expected loss of German Bund by combining PDs and LGD under macro states
def bund_el_from_pd(pd5_DE: float,
                    probabilities: np.ndarray, 
                    base_LGD: float, LGD_scalers: np.ndarray) -> float:
    PDs_states = split_pd_states(np.array([pd5_DE]), probabilities).flatten()
    LGD_states = base_LGD * LGD_scalers
    return float(np.sum(probabilities * PDs_states * LGD_states)) # EL calculation

#%% Per month Monte Carlo engine
# Simulate losses for a give month  by combining state-dependent PD/LGD with a calibrated t-copula
def simulate_month_tcopula(weights_row: pd.Series, pd5_row: pd.Series,
                           probabilities: np.ndarray, 
                           base_LGD: float, LGD_scalers: np.ndarray,
                           n: int, sub_grid: np.ndarray,
                           rng: np.random.Generator, hurdle: float,
                           nu: float,
                           PDs_full: pd.DataFrame,
                           window_months: int = 60,
                           t_anchor: pd.Timestamp | None = None,
                           verbose: bool = True,
                           fixed_lambda: float | None = None,
                           diag_store: list | None = None):
    if t_anchor is None:
        t_anchor = getattr(weights_row, "name", PDs_full.index.max())

    # base active set (positive weight)
    w0 = weights_row.fillna(0.0).copy()
    p0 = pd5_row.fillna(0.0).copy()
    active0 = list(w0.index[w0.values > 0])

    # rolling window estimate (R, nu, lambda) and columns used
    R, nu_hat, lam_hat, used_cols = estimate_corr_and_nu_grid(
        PDs_full, t_anchor, active0, window_months=window_months,
        nu_grid=( 4, 5, 7, 10,15), fixed_lambda=fixed_lambda, diag_store=diag_store
    )

    if len(used_cols) < 1:
        return np.nan, np.nan, {float(s): np.nan for s in sub_grid}, np.nan

    w = w0[used_cols].to_numpy(float)
    pd5 = p0[used_cols].to_numpy(float)
    pd5 = np.clip(pd5, 0.0, 1.0)

    w_sum = w.sum()
    if w_sum <= 0:
        return np.nan, np.nan, {float(s): np.nan for s in sub_grid}, np.nan
    w /= w_sum

    S = len(probabilities)
    N = len(used_cols)

    PDs_SxN = split_pd_states(pd5, probabilities)
    LGD_SxN = lgd_states_matrix(N, base_LGD, LGD_scalers)

    if verbose:
        lam_tag = f"{lam_hat:.6f}" + (" (fixed)" if fixed_lambda is not None else "")
        print(f"Date: {t_anchor:%Y-%m}, nu_used={nu_hat:>5.1f}, lambda={lam_tag}, "
              f"N={N}, used_cols={len(used_cols)}")

    # Simulate macro states + defaults
    states = rng.choice(S, size=n, p=probabilities)
    losses = np.empty(n, dtype=float)

    for s in range(S):
        idx = np.where(states == s)[0]
        if idx.size == 0:
            continue
        U = sample_t_copula_uniforms(idx.size, R, nu=nu_hat, rng=rng)
        default = (U < PDs_SxN[s, :])
        losses[idx] = (default * (w * LGD_SxN[s, :])).sum(axis=1)

    # Pool EL (per pool notional)
    EL_pool = float(losses.mean())
    pool_loss_p99 = float(np.percentile(losses, 99))

    # --- Build tranche EL curves **normalized to tranche notional** ---
    EL_senior_norm = {}
    for s in sub_grid:
        s = float(s)
        ws = 1.0 - s
        el_sen_pool = float(np.mean(np.maximum(losses - s, 0.0)))  # per pool notional
        EL_senior_norm[s] = (el_sen_pool / ws) if ws > 0 else np.nan  # normalize to senior notional

    # Find s* from normalized senior EL
    s_star = np.nan
    for s in sub_grid:
        if EL_senior_norm[float(s)] <= hurdle:
            s_star = float(s)
            break

    # Return normalized senior curve
    return s_star, EL_pool, EL_senior_norm, pool_loss_p99

#%% Run series & build full output
# run over all months 
def run_and_build_tcopula(PDs: pd.DataFrame, final_weights: pd.DataFrame,
                          probabilities, base_LGD, LGD_scalers,
                          n, sub_grid, rng, hurdle, Germany="Germany", s_fixed=0.30,
                          nu: float = 15.0, window_months: int = 60,
                          fixed_lambda: float | None = None, verbose=True,
                          diag_store: list | None = None):
    common_idx  = final_weights.index.intersection(PDs.index)
    common_cols = final_weights.columns.intersection(PDs.columns)
    W = final_weights.loc[common_idx, common_cols]
    P = PDs.loc[common_idx, common_cols]

    rows = []
    for t in W.index:
        s_star, EL_pool, EL_senior_norm_curve, pool_loss_p99 = simulate_month_tcopula(
            W.loc[t], P.loc[t],
            probabilities, base_LGD, LGD_scalers,
            n, sub_grid, rng, hurdle,
            nu=nu, PDs_full=PDs,
            window_months=window_months, t_anchor=t, verbose=verbose,
            fixed_lambda=fixed_lambda, diag_store=diag_store
        )

        # Germany EL (per-bond own-notional)
        pd_de = float(P.loc[t].get(Germany, np.nan))
        EL_bund = np.nan if np.isnan(pd_de) else bund_el_from_pd(
            pd_de, probabilities, base_LGD, LGD_scalers
        )

        # --- Values at s* (normalized to tranche notional) ---
        if np.isnan(s_star):
            EL_sen_opt = np.nan
            EL_jun_opt = np.nan
            senior_share_opt = np.nan
        else:
            EL_sen_opt = EL_senior_norm_curve[float(s_star)]
            # From identity: EL_pool = (1-s)*EL_sen_norm + s*EL_jun_norm
            EL_jun_opt = ((EL_pool - (1.0 - s_star) * EL_sen_opt) / s_star) if s_star > 0 else np.nan
            senior_share_opt = 1.0 - s_star

        # --- Values at fixed s = 0.30 (normalized) ---
        EL_sen_30 = EL_senior_norm_curve.get(float(s_fixed), np.nan)
        EL_jun_30 = ((EL_pool - (1.0 - s_fixed) * EL_sen_30) / s_fixed) if (np.isfinite(EL_sen_30) and s_fixed > 0) else np.nan
        senior_share_30 = 1.0 - s_fixed

        # Safe-asset multipliers unchanged
        w_de = float(W.loc[t].get(Germany, np.nan))
        if np.isnan(w_de) or w_de <= 0.0:
            safe_mult_opt = np.nan
            safe_mult_30  = np.nan
        else:
            safe_mult_opt = senior_share_opt / w_de if not np.isnan(senior_share_opt) else np.nan
            safe_mult_30  = senior_share_30 / w_de

        row = {
            "Month": t,
            "optimal_sub": s_star,
            "EL_sen_optimal": EL_sen_opt,     # normalized senior @ s*
            "EL_jun_optimal": EL_jun_opt,     # normalized junior @ s*
            "EL_sen_30": EL_sen_30,           # normalized senior @ 30%
            "EL_jun_30": EL_jun_30,           # normalized junior @ 30%
            "EL_pool": EL_pool,               # pool EL (per pool notional; kept for context)
            "EL_germany": EL_bund,
            "safe_asset_multiplier_optimal": safe_mult_opt,
            "safe_asset_multiplier_30": safe_mult_30,
            "PoolLoss_p99": pool_loss_p99
        }

        # Store normalized senior checkpoints
        row = add_el_checkpoints(row, EL_senior_norm_curve, levels=(0,10,20,30,40,50,60,70))
        rows.append(row)

    return pd.DataFrame(rows).set_index("Month").sort_index()

#%% Execute main tranche run — Baseline (optimized λ via t-copula likelihood)
print("\n=== BASELINE (optimized λ via t-copula pseudo-likelihood) ===")
tcopula_diag.clear()
results_tranches_t = run_and_build_tcopula(
    PDs, final_weights,
    probabilities, base_LGD, LGD_scalers,
    n, sub_grid, rng, hurdle, Germany=Germany, s_fixed=sub_fixed,
    nu=nu_tcopula, window_months=60, fixed_lambda=None, verbose=True,
    diag_store=tcopula_diag)
print(results_tranches_t.head(10))
diag_df = pd.DataFrame(tcopula_diag).sort_values(["Month","nu"]).reset_index(drop=True)
# Save baseline output
results_tranches_t.to_csv("data /MC_results_tcopula.csv", float_format="%.6f")

#%% Fixed-λ scenario (robustness)
tcopula_diag_fixed.clear()
if fixed_lambda_scenario is not None:
    print(f"\n=== FIXED-λ SCENARIO (λ={fixed_lambda_scenario:.2f}) ===")
    results_tranches_t_fix = run_and_build_tcopula(
        PDs, final_weights,
        probabilities, base_LGD, LGD_scalers,
        n, sub_grid, rng, hurdle, Germany=Germany, s_fixed=sub_fixed,
        nu=nu_tcopula, window_months=60, fixed_lambda=fixed_lambda_scenario, 
        verbose=True, diag_store=tcopula_diag_fixed)
    print(results_tranches_t_fix.head(10))
    diag_df_fixed= pd.DataFrame(tcopula_diag_fixed).sort_values(["Month","nu"]).reset_index(drop=True)
    # Save scenario output
    results_tranches_t_fix.to_csv(
        f"data /MC_results_tcopula_lambda{int(100*fixed_lambda_scenario):03d}.csv",
        float_format="%.6f"
    )

#%% Compare baseline (optimized λ) vs fixed-λ scenario (λ=0.50)
if (fixed_lambda_scenario is not None) and ('results_tranches_t_fix' in locals()):
    # 1) Align and build comparison frame
    metrics = [
        'optimal_sub', 'EL_pool', 'EL_sen_optimal', 'EL_sen_30', 'EL_jun_30',
        'EL_germany', 'safe_asset_multiplier_optimal', 'PoolLoss_p99'
    ]
    base = results_tranches_t[metrics].copy().add_suffix('_opt')
    alt  = results_tranches_t_fix[metrics].copy().add_suffix('_fix')
    cmp_df = base.join(alt, how='inner').sort_index()

    # 2) Compute deltas and helper stats
    for m in metrics:
        cmp_df[f'delta_{m}'] = cmp_df[f'{m}_opt'] - cmp_df[f'{m}_fix']
        cmp_df[f'abs_delta_{m}'] = np.abs(cmp_df[f'delta_{m}'])

    def _summ(m):
        x = cmp_df[f'{m}_opt'].to_numpy(dtype=float)
        y = cmp_df[f'{m}_fix'].to_numpy(dtype=float)
        d = cmp_df[f'delta_{m}'].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y, d = x[mask], y[mask], d[mask]
        if d.size == 0:
            return pd.Series({'N':0})
        mae = float(np.mean(np.abs(d)))
        rmse = float(np.sqrt(np.mean(d**2)))
        mean_d = float(np.mean(d))
        med_d  = float(np.median(d))
        p95abs = float(np.percentile(np.abs(d), 95))
        corr   = float(np.corrcoef(x, y)[0,1]) if x.size>1 else np.nan
        r2     = float(np.maximum(0.0, corr**2)) if np.isfinite(corr) else np.nan
        return pd.Series({
            'N': int(d.size),
            'mean_delta': mean_d,
            'median_delta': med_d,
            'MAE': mae,
            'RMSE': rmse,
            'p95_|delta|': p95abs,
            'corr(opt,fix)': corr,
            'R2': r2
        })

    summary = pd.concat({_m: _summ(_m) for _m in metrics}, axis=1).T
    print("\n=== Baseline (optimized λ) vs Fixed-λ Comparison (λ=%.2f) ===" % fixed_lambda_scenario)
    print(summary.round(6))

    # thresholds tailored to each metric (for a quick sense-check)
    thres = {
        'optimal_sub': 0.01,        # 1% subordination
        'EL_pool': 2e-4,            # 2 bps in EL
        'EL_sen_optimal': 2e-4,
        'EL_sen_30': 2e-4,
        'EL_jun_30': 2e-4,
        'EL_germany': 2e-4,
        'safe_asset_multiplier_optimal': 0.05,  # 5% of DE weight
        'PoolLoss_p99': 0.01        # 1% absolute pool loss
    }
    exceed_rows = []
    for m in metrics:
        d = cmp_df[f'abs_delta_{m}']
        thr = thres[m]
        share = float(np.mean(d > thr)) if d.size else np.nan
        exceed_rows.append({'metric': m, 'threshold': thr, 'share_|delta|>thr': share})
    exceed_tbl = pd.DataFrame(exceed_rows).set_index('metric')
    print("\nShare of months with |delta| above heuristic thresholds:")
    print(exceed_tbl.round(4))

    # 3) Time-series overlays for core metrics
    core = ['EL_pool', 'optimal_sub', 'EL_sen_optimal', 'PoolLoss_p99']
    plt.figure(figsize=(12, 8))
    for i, m in enumerate(core, 1):
        ax = plt.subplot(2, 2, i)
        ax.plot(cmp_df.index, cmp_df[f'{m}_opt'], label=f'{m} (opt λ)', linewidth=1.6)
        ax.plot(cmp_df.index, cmp_df[f'{m}_fix'], label=f'{m} (λ={fixed_lambda_scenario:.2f})', linewidth=1.2, alpha=0.8)
        ax.set_title(m.replace('_',' ').title()); ax.grid(True, alpha=0.3)
        if i in (1,2): ax.legend()
    plt.tight_layout(); plt.show()

    # 4) Scatter vs 45° line for the same core metrics
    """  plt.figure(figsize=(12, 8))
    for i, m in enumerate(core, 1):
        ax = plt.subplot(2, 2, i)
        x = cmp_df[f'{m}_opt'].to_numpy()
        y = cmp_df[f'{m}_fix'].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        ax.scatter(x, y, s=18, alpha=0.7)
        if x.size:
            lo = np.nanmin([x.min(), y.min()])
            hi = np.nanmax([x.max(), y.max()])
            ax.plot([lo, hi], [lo, hi], '--', alpha=0.6)
        ax.set_xlabel(f'{m} (opt λ)'); ax.set_ylabel(f'{m} (λ={fixed_lambda_scenario:.2f})')
        ax.set_title(f'{m}: opt vs fixed'); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show() """

    # 5) Delta time-series (core metrics)
    plt.figure(figsize=(12, 8))
    for i, m in enumerate(core, 1):
        ax = plt.subplot(2, 2, i)
        ax.plot(cmp_df.index, cmp_df[f'delta_{m}'], linewidth=1.5)
        ax.axhline(0.0, ls='--', alpha=0.6)
        ax.set_title(f'Delta {m} (opt - fixed)'); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    # 6) Mean Absolute Delta by metric (bar chart)
    mad_vals = [cmp_df[f'abs_delta_{m}'].mean() for m in metrics]
    plt.figure(figsize=(11, 4))
    plt.bar([m for m in metrics], mad_vals)
    plt.title('Mean |delta| by metric (opt λ minus fixed λ)')
    plt.xticks(rotation=30, ha='right'); plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout(); plt.show()

else:
    print("\n[Compare] Skipped: fixed-λ scenario not run or results not found.")

#%% Convergence Test & Diagnostics (accepts fixed_lambda for scenario checks)
def convergence_test_tcopula(PDs, final_weights,
                             probabilities, base_LGD, LGD_scalers,
                             sub_grid, rng, hurdle, Germany="Germany", s_fixed=0.30,
                             test_date="2015-12", n_values=None, repeats=10, base_seed=7,
                             nu: float = 7.0, window_months: int = 60,
                             fixed_lambda: float | None = None):
    if n_values is None:
        n_values = [1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000, 300_000, 400_000, 500_000, 1_000_000]

    test_timestamp = pd.Timestamp(test_date)
    common_idx = final_weights.index.intersection(PDs.index)
    if test_timestamp not in common_idx:
        closest_date = min(common_idx, key=lambda x: abs(x - test_timestamp))
        print(f"Test date {test_date} not found. Using closest date: {closest_date}")
        test_timestamp = closest_date

    common_cols = final_weights.columns.intersection(PDs.columns)
    w_test = final_weights.loc[test_timestamp, common_cols]
    pd_test = PDs.loc[test_timestamp, common_cols]

    rows = []
    tag = " (fixed λ)" if fixed_lambda is not None else ""
    print(f"Running convergence test (t-copula{tag})...")
    print(f"Test date: {test_timestamp}")
    print(f"Number of active countries: {(w_test.fillna(0) > 0).sum()}\n")

    for i, n_ in enumerate(n_values):
        print(f"Testing n={n_:,} ({i+1}/{len(n_values)})...")
        bucket = []
        for r in range(repeats):
            rng_local = np.random.default_rng(base_seed + r + 7919*n_)
            s_star, EL_pool, EL_senior_norm_curve, pool_loss_p99 = simulate_month_tcopula(
                w_test, pd_test,
                probabilities, base_LGD, LGD_scalers,
                n_, sub_grid, rng_local, hurdle,
                nu=nu, PDs_full=PDs, window_months=window_months,
                t_anchor=test_timestamp, verbose=False, fixed_lambda=fixed_lambda
            )

            # Normalized senior/junior at s* and 30%
            EL_sen_opt = EL_senior_norm_curve[float(s_star)] if not np.isnan(s_star) else np.nan
            EL_jun_opt = ((EL_pool - (1.0 - s_star) * EL_sen_opt) / s_star) if (not np.isnan(s_star) and s_star > 0) else np.nan
            EL_sen_30  = EL_senior_norm_curve.get(float(s_fixed), np.nan)
            EL_jun_30  = ((EL_pool - (1.0 - s_fixed) * EL_sen_30) / s_fixed) if (np.isfinite(EL_sen_30) and s_fixed > 0) else np.nan

            # Germany (own-notional)
            pd_de = float(pd_test.get(Germany, np.nan))
            EL_bund = np.nan if np.isnan(pd_de) else bund_el_from_pd(
                pd_de, probabilities, base_LGD, LGD_scalers
            )

            bucket.append({
                'optimal_sub': s_star,
                'EL_pool': EL_pool,
                'EL_sen_optimal': EL_sen_opt,   # normalized
                'EL_jun_optimal': EL_jun_opt,   # normalized
                'EL_sen_30': EL_sen_30,         # normalized
                'EL_jun_30': EL_jun_30,         # normalized
                'EL_germany': EL_bund,
                'PoolLoss_p99': pool_loss_p99
            })

        def mean_se(key):
            arr = np.array([b[key] for b in bucket], dtype=float)
            arr = arr[~np.isnan(arr)]
            if arr.size == 0:
                return np.nan, np.nan
            mu = float(arr.mean())
            se = float(arr.std(ddof=1)/np.sqrt(arr.size)) if arr.size > 1 else 0.0
            return mu, se

        mu_s,   se_s   = mean_se('optimal_sub')
        mu_ep,  se_ep  = mean_se('EL_pool')
        mu_es,  se_es  = mean_se('EL_sen_optimal')
        mu_p99, se_p99 = mean_se('PoolLoss_p99')
        mu_e30, se_e30 = mean_se('EL_sen_30')
        mu_j30, se_j30 = mean_se('EL_jun_30')
        mu_g,   se_g   = mean_se('EL_germany')

        rows.append({
            'n_simulations': n_,
            # means
            'optimal_sub': mu_s,
            'EL_pool': mu_ep,
            'EL_sen_optimal': mu_es,   # normalized
            'EL_sen_30': mu_e30,       # normalized
            'EL_jun_30': mu_j30,       # normalized
            'EL_germany': mu_g,
            'PoolLoss_p99': mu_p99,
            # SEs
            'optimal_sub_se': se_s,
            'EL_pool_se': se_ep,
            'EL_sen_optimal_se': se_es,
            'EL_sen_30_se': se_e30,
            'EL_jun_30_se': se_j30,
            'EL_germany_se': se_g,
            'PoolLoss_p99_se': se_p99,
            # relative MC error %
            'EL_pool_mc_error_pct': (se_ep / mu_ep * 100) if (mu_ep not in (0, np.nan)) else np.nan,
            'EL_sen_optimal_mc_error_pct': (se_es / mu_es * 100) if (mu_es not in (0, np.nan)) else np.nan,
            'PoolLoss_p99_mc_error_pct': (se_p99 / mu_p99 * 100) if (mu_p99 not in (0, np.nan)) else np.nan,
            'repeats': repeats
        })

    return pd.DataFrame(rows)


def plot_convergence(conv_results, metrics_to_plot=None):
    if metrics_to_plot is None:
        metrics_to_plot = ['EL_pool', 'optimal_sub', 'EL_sen_optimal', 'PoolLoss_p99']

    available_metrics = [m for m in metrics_to_plot
                         if m in conv_results.columns and not conv_results[m].isna().all()]
    if not available_metrics:
        print("No valid metrics to plot!")
        return

    n_metrics = min(len(available_metrics), 4)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, metric in enumerate(available_metrics[:4]):
        ax = axes[i]
        n_sims = conv_results['n_simulations'].values
        y = conv_results[metric].values
        se_col = f"{metric}_se"
        has_se = se_col in conv_results.columns and not conv_results[se_col].isna().all()

        ax.plot(n_sims, y, 'o-', linewidth=2, markersize=6, label='Mean')
        if has_se:
            se = conv_results[se_col].values
            ax.fill_between(n_sims, y - 2*se, y + 2*se, alpha=0.20, linewidth=0, label='±2·SE')

        ax.set_xscale('log')
        ax.set_xlabel('Number of Simulations')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Convergence: {metric}')
        ax.grid(True, alpha=0.3)
        ax.legend()

        if not np.isnan(y[-1]):
            ax.axhline(y=y[-1], color='red', linestyle='--', alpha=0.5)

    for i in range(n_metrics, 4):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig

def analyze_convergence_stability(conv_results, metric='EL_pool', tolerance=0.001):
    if metric not in conv_results.columns:
        return f"Metric '{metric}' not found in results"
    values = conv_results[metric].dropna().values
    n_sims = conv_results.loc[conv_results[metric].notna(), 'n_simulations'].values
    if len(values) < 2:
        return "Not enough valid values for stability analysis"
    rel_changes = np.abs(np.diff(values)) / np.clip(np.abs(values[:-1]), 1e-12, None)
    stable_idx = np.where(rel_changes < tolerance)[0]
    analysis = {
        'metric': metric,
        'final_value': values[-1],
        'max_relative_change': rel_changes.max(),
        'mean_relative_change': rel_changes.mean(),
        'relative_changes': rel_changes,
        'n_simulations': n_sims,
        'tolerance': tolerance
    }
    if len(stable_idx) > 0:
        first_stable = stable_idx[0] + 1
        analysis['converged_at_n'] = int(n_sims[first_stable])
        analysis['converged'] = True
    else:
        analysis['converged'] = False
        analysis['converged_at_n'] = None
    return analysis

def analyze_convergence_ci(conv_results, metric='EL_sen_optimal', conf=0.95,
                           abs_tol=None, rel_tol=None):
    if metric not in conv_results.columns or f"{metric}_se" not in conv_results.columns:
        return {"ok": False, "reason": f"{metric} or its SE not found"}
    df = conv_results.sort_values('n_simulations').copy()
    mu = df[metric].to_numpy()
    se = df[f"{metric}_se"].to_numpy()
    ns = df['n_simulations'].to_numpy()

    z = 1.96 if abs(conf - 0.95) < 1e-9 else 1.96
    ci_half = z * se

    ok_idx = []
    for i in range(len(df)):
        cond_abs = (abs_tol is None) or (ci_half[i] <= abs_tol)
        cond_rel = (rel_tol is None) or (mu[i] != 0 and ci_half[i] <= rel_tol * abs(mu[i]))
        cond_ovl = True
        if i >= 1:
            cond_ovl = abs(mu[i] - mu[i-1]) <= max(ci_half[i], ci_half[i-1])
        if cond_abs and cond_rel and cond_ovl:
            ok_idx.append(i)

    if ok_idx:
        return {
            "ok": True,
            "converged_at_n": int(ns[ok_idx[0]]),
            "last_mu": float(mu[-1]),
            "last_ci_half": float(ci_half[-1]),
            "repeats": int(df['repeats'].iloc[-1]),
            "needed_n": int(ns[ok_idx[0]])
        }

    # projection when not yet converged: CI ∝ 1/√n  =>  n_needed = n_last * (ci_last/target)^2
    n_last = int(ns[-1])
    mu_last = float(mu[-1])
    ci_last = float(ci_half[-1])

    targets = []
    if abs_tol is not None:
        targets.append(abs_tol)
    if rel_tol is not None and mu_last != 0:
        targets.append(rel_tol * abs(mu_last))
    if not targets:
        return {
            "ok": False, "converged_at_n": None,
            "last_mu": mu_last, "last_ci_half": ci_last,
            "repeats": int(df['repeats'].iloc[-1]),
            "needed_n": None
        }
    target_half = min(targets)
    if target_half <= 0 or ci_last == 0:
        needed_n = n_last
    else:
        needed_n = int(np.ceil(n_last * (ci_last / target_half) ** 2))

    return {
        "ok": False, "converged_at_n": None,
        "last_mu": mu_last, "last_ci_half": ci_last,
        "repeats": int(df['repeats'].iloc[-1]),
        "needed_n": needed_n
    }

def stability_in_probability_sstar(PDs, final_weights, month, n_paths, repeats=20, base_seed=7):
    common_cols = final_weights.columns.intersection(PDs.columns)
    w_row = final_weights.loc[month, common_cols]
    p_row = PDs.loc[month, common_cols]
    s_vals = []
    for r in range(repeats):
        rng_local = np.random.default_rng(base_seed + r + 104729*n_paths)
        s_star, EL_pool, EL_senior_curve, pool_loss_p99 = simulate_month_tcopula(
            w_row, p_row,
            probabilities, base_LGD, LGD_scalers,
            n_paths, sub_grid, rng_local, hurdle,
            nu=nu_tcopula, PDs_full=PDs, window_months=win_months, t_anchor=month, verbose=False
        )
        s_vals.append(s_star)
    s_vals = np.array(s_vals, dtype=float)
    if np.all(np.isnan(s_vals)):
        return {"modal_s": np.nan, "probability": 0.0, "repeats": repeats}
    s_clean = s_vals[~np.isnan(s_vals)]
    if s_clean.size == 0:
        return {"modal_s": np.nan, "probability": 0.0, "repeats": repeats}
    uniq, counts = np.unique(s_clean, return_counts=True)
    idx = np.argmax(counts)
    modal_s = float(uniq[idx])
    prob = float(counts[idx] / s_clean.size)
    return {"modal_s": modal_s, "probability": prob, "repeats": repeats}

#%% Convergence Test (t-copula, baseline with optimized λ)
print("="*50)
print("MONTE CARLO CONVERGENCE TEST — t-copula (baseline = optimized λ)")
print("="*50)
conv_results_t = convergence_test_tcopula(
    PDs, final_weights,
    probabilities, base_LGD, LGD_scalers,
    sub_grid, rng, hurdle, Germany, sub_fixed,
    test_date="2015-12",
    n_values=[1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000, 300_000, 400_000, 500_000, 1_000_000],
    repeats=12, base_seed=7,
    nu=nu_tcopula, window_months=win_months, fixed_lambda=None
)
print("\nConvergence Results (means ± SE):")
print("-" * 80)
print(conv_results_t.round(6))

print("\nGenerating convergence plots...")
plot_convergence(conv_results_t, metrics_to_plot = ['EL_pool', 'optimal_sub', 'EL_sen_optimal', 'PoolLoss_p99'])

print("\nStability Analysis (relative-change rule):")
print("-" * 40)
for metric, tol in [('EL_pool', 0.001), ('EL_sen_optimal', 0.001), ('PoolLoss_p99', 0.02)]:
    if metric in conv_results_t.columns:
        stability = analyze_convergence_stability(conv_results_t, metric, tolerance=tol)
        if isinstance(stability, dict):
            print(f"\n{metric}:")
            print(f"  Final value: {stability['final_value']:.6f}")
            print(f"  Max relative change: {stability['max_relative_change']:.4f}")
            print(f"  Mean relative change: {stability['mean_relative_change']:.4f}")
            if stability['converged']:
                print(f"  Converged at n = {stability['converged_at_n']:,} (tol={tol})")
            else:
                print(f"  Not yet converged (tol={tol})")

print("\nCI-based Analysis (95% CI):")
print("-" * 40)
ci_checks = [
    ('EL_sen_optimal', 0.00020, None),  # 2 bps absolute for EL metrics
    ('EL_pool',        0.00020, None),  # 2 bps absolute for EL metrics
    ('PoolLoss_p99',   None,    0.02)   # 2% relative for tails
]
ci_decisions = []
for metric, abs_tol, rel_tol in ci_checks:
    if metric in conv_results_t.columns and f"{metric}_se" in conv_results_t.columns:
        res = analyze_convergence_ci(conv_results_t, metric=metric, conf=0.95,
                                     abs_tol=abs_tol, rel_tol=rel_tol)
        ci_decisions.append((metric, res))
        print(f"\n{metric}:")
        if res["ok"]:
            print(f"  Converged at n = {res['converged_at_n']:,}")
        else:
            need_txt = "unknown" if res["needed_n"] is None else f"{res['needed_n']:,}"
            print("  Not yet converged by CI rule.")
            print(f"  Projected n to satisfy CI tolerances ≈ {need_txt}")
        print(f"  Last mean = {res['last_mu']:.6f}, last 95% half-CI = {res['last_ci_half']:.6f}, repeats = {res['repeats']}")

# Final CI-based recommendation across metrics (max over needed n so all pass)
tested_max_n = int(conv_results_t['n_simulations'].max())
per_metric_needed = []
binding_metrics = []
for metric, res in ci_decisions:
    if res.get("ok", False):
        per_metric_needed.append(int(res["converged_at_n"]))
    elif res.get("needed_n") is not None:
        per_metric_needed.append(int(res["needed_n"]))
    else:
        per_metric_needed.append(tested_max_n)

if per_metric_needed:
    recommended_n = max(per_metric_needed)
    for metric, res in ci_decisions:
        if (res.get('ok', False) and res.get('converged_at_n') == recommended_n) or \
           (not res.get('ok', False) and res.get('needed_n') == recommended_n):
            binding_metrics.append(metric)
    print("\n" + "="*70)
    print(f"Based on the CI approach, I recommend n = {recommended_n:,} simulations "
          f"(binding metric(s): {', '.join(binding_metrics)}).")
    print("="*70)
else:
    print("\nNo CI-based recommendation could be formed (missing SEs or metrics).")

#%% Convergence Test for the fixed-λ scenario
if fixed_lambda_scenario is not None:
    print("="*50)
    print(f"MONTE CARLO CONVERGENCE TEST — t-copula (fixed λ={fixed_lambda_scenario:.2f})")
    print("="*50)
    conv_results_t_fix = convergence_test_tcopula(
        PDs, final_weights,
        probabilities, base_LGD, LGD_scalers,
        sub_grid, rng, hurdle, Germany, sub_fixed,
        test_date="2015-12",
        n_values=[1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000, 300_000, 400_000, 500_000, 1_000_000],
        repeats=12, base_seed=7,
        nu=nu_tcopula, window_months=win_months, fixed_lambda=fixed_lambda_scenario
    )
    print("\nConvergence Results (means ± SE):")
    print("-" * 80)
    print(conv_results_t_fix.round(6))
    print("\nGenerating convergence plots...")
    plot_convergence(conv_results_t_fix)

print("\nRecommendation:")
final_n = conv_results_t['n_simulations'].iloc[-1]
print(
    f"Based on this test, your current n = {n:,} appears adequate for the mean metrics if they converged by CI and relative rules."
    if final_n >= n else
    f"Potentially insufficient. Consider increasing to at least {final_n:,} or until CI-based criteria are met."
)

#%% PLOTS -------------------



#%% Plot choice of degrees of freedom
#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler

# ===============================
# Global minimalist B&W style
# ===============================
plt.rcParams.update({
    # Typography
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",

    # Figure & Axes
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 0.9,

    # Ticks
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.labelsize": 12,  # tick labels 12 pt
    "ytick.labelsize": 12,  # tick labels 12 pt

    # Grid (horizontal only; subtle dotted)
    "grid.color": "0.85",
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,

    # Monochrome lines
    "axes.prop_cycle": cycler("color", ["black"]),
})

# ===============================
# Helpers
# ===============================
def _prep_month_col(df, month_col="Month"):
    out = df.copy()
    out[month_col] = pd.to_datetime(out[month_col]).dt.to_period("M").dt.to_timestamp("M")
    return out

def _delta_to_runner_up(g):
    a = np.sort(g["AIC"].values)
    return a[1] - a[0] if len(a) >= 2 else np.nan

def _ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

# ===============================
# Plot suite (monochrome)
# ===============================
def plot_diag_suite(diag_df, tag="optimized"):
    """
    Creates and saves:
      (a) step plot of AIC-minimizing ν over time
      (c) ΔAIC to runner-up over time (reference lines at 2, 4, 7, 10)
      (d) λ̂ over time for the winning ν
    All figures are black & white, journal-style.
    """
    df = _prep_month_col(diag_df, "Month")

    # --- (a) Chosen ν over time (AIC winner) ---
    winner_idx = df.groupby("Month")["AIC"].idxmin()
    winner = df.loc[winner_idx].sort_values("Month")

    fig, ax = plt.subplots(figsize=(10.8, 4.2))  # ~5:3
    ax.step(winner["Month"], winner["nu"], where="mid", linewidth=1.6)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel(r"$\nu$ (df)", fontsize=12)
    ax.yaxis.grid(True); ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)
    # Minimal title at 12 pt (optional)
    ax.set_title("Selected t-copula degrees of freedom (AIC-minimizing)", fontsize=12)
    fig.tight_layout()
    path_a = f"FinalFigures/diag_{tag}_a_nu_over_time.png"
    _ensure_dir(path_a); fig.savefig(path_a, dpi=300, bbox_inches="tight")

    # --- (c) Strength of selection: ΔAIC to 2nd best (per month) ---
    delta2 = df.groupby("Month").apply(_delta_to_runner_up).rename("deltaAIC_runnerup").reset_index()
    fig, ax = plt.subplots(figsize=(10.8, 4.0))
    ax.plot(delta2["Month"], delta2["deltaAIC_runnerup"], marker="o", markersize=3.5,
            linewidth=1.4, linestyle="-", label=r"$\Delta$AIC to runner-up")

    # Reference lines (monochrome, different dash styles)
    ref_lines = [
        (2,  (2, 2), "ΔAIC = 2"),
        #(4,  (6, 2), "ΔAIC = 4"),
        #(7,  (6, 3, 2, 3), "ΔAIC = 7"),
        #(10, (1, 2), "ΔAIC = 10"),
    ]
    
    for y, dash, lab in ref_lines:
        ln = ax.axhline(y, color="black", linewidth=1.0, linestyle="--", label=lab)
        ln.set_dashes(dash)

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel(r"$\Delta$AIC", fontsize=12)
    ax.yaxis.grid(True); ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    leg = ax.legend(loc="upper left", frameon=True, fontsize=10, facecolor="white", edgecolor="black")
    leg.get_frame().set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_title("Evidence for selected ν: ΔAIC to runner-up", fontsize=12)
    fig.tight_layout()
    path_c = f"FinalFigures/diag_{tag}_c_deltaAIC_runnerup.png"
    _ensure_dir(path_c); fig.savefig(path_c, dpi=300, bbox_inches="tight")

    # --- (d) λ̂ path for the winning ν each month ---
    fig, ax = plt.subplots(figsize=(10.8, 4.0))
    ax.plot(winner["Month"], winner["lambda_hat"], linewidth=1.6, linestyle="-")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel(r"$\hat{\lambda}$", fontsize=12)
    ax.yaxis.grid(True); ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_title(r"Estimated shrinkage $\hat{\lambda}$ over time", fontsize=12)
    fig.tight_layout()
    path_d = f"FinalFigures/diag_{tag}_d_lambda_hat.png"
    _ensure_dir(path_d); fig.savefig(path_d, dpi=300, bbox_inches="tight")

    return {"a": path_a, "c": path_c, "d": path_d}

# =========================================
# Optional: monochrome heatmap (commented)
# To use, uncomment this block.
# Note: sticks to B&W using a grayscale colormap.
# =========================================
"""
def plot_diag_heatmap_monochrome(diag_df, tag="optimized"):
    df = _prep_month_col(diag_df, "Month")
    pv = df.pivot_table(index="Month", columns="nu", values="AIC", aggfunc="min")
    pv_delta = pv.sub(pv.min(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(pv_delta.T.values, aspect="auto", cmap="Greys", interpolation="nearest")
    ax.set_xlabel("Month", fontsize=12); ax.set_ylabel(r"$\nu$", fontsize=12)
    ax.tick_params(axis="both", labelsize=12)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_title(r"$\Delta$AIC by Month and $\nu$ (row-min subtracted)", fontsize=12)
    fig.tight_layout()
    path_b = f"FinalFigures/diag_{tag}_b_heatmap.png"
    _ensure_dir(path_b); fig.savefig(path_b, dpi=300, bbox_inches="tight")
    return path_b
"""

# ===============================
# Driver: run for both datasets
# ===============================
PLOT = True  # set True to generate and save figures
if PLOT:
    # Assumes you already have `diag_df` and `diag_df_fixed` in memory
    paths_opt   = plot_diag_suite(diag_df, tag="optimized")
    paths_fixed = plot_diag_suite(diag_df_fixed, tag="fixed")
    print("Saved optimized:", paths_opt)
    print("Saved fixed    :", paths_fixed)
#%% FIGURE 8

#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from cycler import cycler

# ===============================
# Global minimalist B&W style
# ===============================
plt.rcParams.update({
    # Typography
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",

    # Axes & figure
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 0.9,

    # Ticks
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,

    # Grid (horizontal only; subtle dotted)
    "grid.color": "0.85",
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,

    # Monochrome lines
    "axes.prop_cycle": cycler("color", ["black"]),
})

# Mid-gray fill for bars (no outline)
BAR_GRAY = "#7f7f7f"

# ----------------------------------------------------------
# Helper: per-country Expected Loss with NO tranching (own-notional)
# Uses model functions: split_pd_states() and lgd_states_matrix()
# ----------------------------------------------------------
def compute_country_el_no_tranching(p_row: pd.Series,
                                    probabilities: np.ndarray,
                                    base_LGD: float,
                                    LGD_scalers: np.ndarray) -> pd.Series:
    """
    p_row: Series of baseline PDs at the chosen month (decimals or percents<=100).
    Returns EL per country (same units as PD*LGD; if PD decimals & LGD in [0,1], EL is decimal).
    """
    p = p_row.astype(float).to_numpy(dtype=float)
    PDs_SxN = split_pd_states(p, probabilities)                 # shape (S, N)
    LGD_SxN = lgd_states_matrix(PDs_SxN.shape[1], base_LGD, LGD_scalers)  # (S, N)
    EL = (probabilities.reshape(-1, 1) * PDs_SxN * LGD_SxN).sum(axis=0)   # (N,)
    return pd.Series(EL, index=p_row.index)

# ----------------------------------------------------------
# Plot: country ELs (no tranching) + pure pooling line
# ----------------------------------------------------------
def plot_country_el_no_tranching(PDs: pd.DataFrame,
                                 final_weights: pd.DataFrame,
                                 probabilities: np.ndarray,
                                 base_LGD: float,
                                 LGD_scalers: np.ndarray,
                                 ref_date_str: str = "2024-12",
                                 name_to_code: dict | None = None,
                                 save_dir: str = "FinalFigures",
                                 bar_color: str = BAR_GRAY):
    """
    Creates a bar chart of per-country expected losses (no tranching), sorted ascending,
    with a bold pure-pooling line (weighted average by portfolio weights).
    """
    if name_to_code is None:
        name_to_code = {
            "Germany":"DE","Netherlands":"NL","Luxembourg":"LU","Austria":"AT","Finland":"FI",
            "France":"FR","Belgium":"BE","Estonia":"EE","Slovakia":"SK","Ireland":"IE",
            "Latvia":"LV","Lithuania":"LT","Malta":"MT","Slovenia":"SI","Spain":"ES",
            "Italy":"IT","Portugal":"PT","Cyprus":"CY","Greece":"GR", "Croatia":"HR",
        }

    # --- choose reference month (closest available if exact missing) ---
    ref_date = pd.Timestamp(ref_date_str).to_period("M").to_timestamp("M")
    common_idx = final_weights.index.intersection(PDs.index)
    if ref_date not in common_idx:
        if len(common_idx) == 0:
            raise ValueError("No common dates between PDs and final_weights.")
        ref_date = min(common_idx, key=lambda x: abs(x - ref_date))

    # --- rows for that month ---
    common_cols = final_weights.columns.intersection(PDs.columns)
    if len(common_cols) == 0:
        raise ValueError("No common country columns between PDs and final_weights.")

    w_row = final_weights.loc[ref_date, common_cols].astype(float)
    p_row = PDs.loc[ref_date, common_cols].astype(float)

    # Active = weight > 0
    active_mask = (w_row.fillna(0.0) > 0.0)

    # --- per-country ELs (no tranching) via model ---
    el_country = compute_country_el_no_tranching(p_row, probabilities, base_LGD, LGD_scalers)

    # keep only active names and drop NaNs
    el_active = el_country[active_mask].dropna()

    # --- pooling line: weighted average EL with portfolio weights ---
    w_active = w_row[active_mask].fillna(0.0)
    w_sum = w_active.sum()
    w_active = (w_active / w_sum) if (w_sum > 0) else w_active
    pooling_line = float((el_active * w_active.loc[el_active.index]).sum())

    # --- x-axis labels (2-letter codes where known) ---
    x_labels = [name_to_code.get(c, c) for c in el_active.index]

    # --- sort ascending (paper style) ---
    order = np.argsort(el_active.values)
    vals = el_active.values[order]
    countries_sorted = [x_labels[i] for i in order]

    # --- percent axis if PD inputs are decimals (common case) ---
    is_decimal = (np.nanmax(p_row.values) <= 1.0) and (0.0 <= base_LGD <= 1.5) and (np.nanmax(LGD_scalers) <= 2.0)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(12, 6.2))

    # Convert expected loss values to percentage scale
    vals_pct = vals * 100
    pooling_line_pct = pooling_line * 100

    # Bars: solid mid-gray fill, no outline
    ax.bar(
        countries_sorted, vals_pct,
        color=bar_color, linewidth=0,
        label="Sovereign bonds with no tranching"
    )

    # Horizontal "Pure pooling" line (bold black)
    ax.axhline(pooling_line_pct, color="black", linewidth=2.0, label="Pure pooling")

    # Labels
    ax.set_xlabel("Country", fontsize=12)
    ax.set_ylabel("Expected loss rate (%)", fontsize=12)

    # Format y-axis: integers only (no decimals, no % sign)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y):d}%"))

    # X ticks
    ax.set_xticklabels(countries_sorted, rotation=90)

    # Grid: horizontal only
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)

    # Minimal frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend: thin border, inside
    leg = ax.legend(loc="upper left", frameon=True, fontsize=10,
                    facecolor="white", edgecolor="black")
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()



    # Save
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"EL_ByCountry_NoTranching_{ref_date:%Y-%m}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")

    return fig, ax, out_path


# --------------------------------
# Do NOT plot unless you set True
# --------------------------------
PLOT = True
if PLOT:
    fig, ax, path = plot_country_el_no_tranching(
        PDs, final_weights,
        probabilities=probabilities,
        base_LGD=base_LGD,
        LGD_scalers=LGD_scalers,
        ref_date_str="2024-12",
    )
    print("Saved:", path)


#%% 
#FIGURE 10
# -*- coding: utf-8 -*-
# Journal-style, monochrome plotting of Expected Loss bars (countries + tranches + pool)
# Tailored to your t-copula model. This version:
#  - Plots immediately
#  - Tranches @ 30% = solid black
#  - Y-ticks spaced 5% apart (5,10,15,20,25,30)
#  - X/Y tick marks made thinner for cleaner academic look

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Patch
from cycler import cycler

# ============================================================
# Global academic B&W style (Computer Modern, dotted y-grid)
# ============================================================
plt.rcParams.update({
    # Typography
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",

    # Figure & Axes
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 0.9,

    # Ticks
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.major.size": 3.5,   # smaller tick length
    "ytick.major.size": 3.5,
    "xtick.major.width": 0.6,  # thinner ticks
    "ytick.major.width": 0.6,

    # Grid (horizontal only, subtle dotted)
    "grid.color": "0.85",
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,

    # Monochrome lines
    "axes.prop_cycle": cycler("color", ["black"]),
})

# ============================================================
# Colors / hatches
# ============================================================
LIGHT_GRAY = "#bfbfbf"  # EL of countries
BAR_GRAY   = "#7f7f7f"  # Tranches at optimal subordination
POOL_FACE  = "white"    # Pool EL face
POOL_EDGE  = "black"
POOL_HATCH = "///"
SUB30_FACE = "black"    # Tranches @ 30% now solid black
SUB30_EDGE = "black"
SUB30_HATCH = None

# ============================================================
# Helper: per-country Expected Loss with NO tranching (own-notional)
# ============================================================
def compute_country_el_no_tranching(p_row: pd.Series,
                                    probabilities: np.ndarray,
                                    base_LGD: float,
                                    LGD_scalers: np.ndarray) -> pd.Series:
    p = p_row.astype(float).to_numpy(dtype=float)
    PDs_SxN = split_pd_states(p, probabilities)
    LGD_SxN = lgd_states_matrix(PDs_SxN.shape[1], base_LGD, LGD_scalers)
    EL = (probabilities.reshape(-1, 1) * PDs_SxN * LGD_SxN).sum(axis=0)
    return pd.Series(EL, index=p_row.index)

# ============================================================
# Main plotting function
# ============================================================
def plot_expected_loss_bars_tcopula(PDs: pd.DataFrame,
                                    final_weights: pd.DataFrame,
                                    results_tranches: pd.DataFrame,
                                    probabilities: np.ndarray,
                                    base_LGD: float,
                                    LGD_scalers: np.ndarray,
                                    ref_date_str: str = "2020-01",
                                    cap_pct: float = 30.0,
                                    name_to_code: dict | None = None,
                                    figsize=(14, 6.0)):
    if name_to_code is None:
        name_to_code = {
            "Germany": "DE", "Netherlands": "NL", "Luxembourg": "LU", "Austria": "AT", "Finland": "FI",
            "France": "FR", "Belgium": "BE", "Estonia": "EE", "Slovakia": "SK", "Ireland": "IE",
            "Latvia": "LV", "Lithuania": "LT", "Malta": "MT", "Slovenia": "SI", "Spain": "ES",
            "Italy": "IT", "Portugal": "PT", "Cyprus": "CY", "Greece": "GR", "Croatia": "HR",
        }

    ref_want = pd.Timestamp(ref_date_str).to_period("M").to_timestamp("M")
    common_idx = final_weights.index.intersection(PDs.index)
    if ref_want not in common_idx:
        if len(common_idx) == 0:
            raise ValueError("No common dates between PDs and final_weights.")
        ref_have = min(common_idx, key=lambda x: abs(x - ref_want))
        print(f"[Info] Requested {ref_want.date()} not found. Using closest: {ref_have.date()}")
    else:
        ref_have = ref_want

    common_cols = final_weights.columns.intersection(PDs.columns)
    if len(common_cols) == 0:
        raise ValueError("No common country columns between PDs and final_weights.")

    w_row = final_weights.loc[ref_have, common_cols].astype(float)
    p_row = PDs.loc[ref_have, common_cols].astype(float)
    active_mask = (w_row.fillna(0.0) > 0.0)

    el_country = compute_country_el_no_tranching(p_row, probabilities, base_LGD, LGD_scalers)
    el_active = el_country[active_mask].copy()
    el_active = el_active[np.isfinite(el_active.values)]

    country_labels = [name_to_code.get(c, c) for c in el_active.index]

    if results_tranches.empty:
        raise ValueError("results_tranches is empty; nothing to plot.")
    r_idx = ref_have if ref_have in results_tranches.index else min(results_tranches.index, key=lambda x: abs(x - ref_have))
    row_r = results_tranches.loc[r_idx]

    EL_pool_val    = float(row_r.get("EL_pool", np.nan))
    EL_sen_opt_val = float(row_r.get("EL_sen_optimal", np.nan))
    EL_jun_opt_val = float(row_r.get("EL_jun_optimal", np.nan))
    EL_sen_30_val  = float(row_r.get("EL_sen_30", np.nan))
    EL_jun_30_val  = float(row_r.get("EL_jun_30", np.nan))

    items = []
    if len(el_active):
        items += [(lab, val, "country") for lab, val in zip(country_labels, el_active.values)]
    items += [("Senior (opt)", EL_sen_opt_val, "opt"),
              ("Junior (opt)", EL_jun_opt_val, "opt"),
              ("Senior (30%)", EL_sen_30_val,  "sub30"),
              ("Junior (30%)", EL_jun_30_val,  "sub30"),
              ("Pool",         EL_pool_val,    "pool")]

    items = [(lab, val, tag) for (lab, val, tag) in items if np.isfinite(val)]
    if not items:
        raise ValueError("No finite EL values to plot.")
    items.sort(key=lambda t: t[1])

    labels_sorted = [t[0] for t in items]
    values_sorted = np.array([t[1] for t in items], dtype=float) * 100.0

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axisbelow(True)

    x_locs = np.arange(len(labels_sorted))
    width = 0.8

    for i, (lab, v_dec, tag) in enumerate(items):
        v_pct = float(v_dec) * 100.0
        draw_height = min(v_pct, cap_pct)

        if tag == "country":
            facecolor = LIGHT_GRAY
            edgecolor = "none"
            hatch     = None
            lw        = 0.0
        elif tag == "opt":
            facecolor = BAR_GRAY
            edgecolor = "none"
            hatch     = None
            lw        = 0.0
        elif tag == "sub30":
            facecolor = SUB30_FACE
            edgecolor = SUB30_EDGE
            hatch     = SUB30_HATCH
            lw        = 0.0
        elif tag == "pool":
            facecolor = POOL_FACE
            edgecolor = POOL_EDGE
            hatch     = POOL_HATCH
            lw        = 1.0
        else:
            facecolor = "white"
            edgecolor = "black"
            hatch     = None
            lw        = 1.0

        ax.bar(
            x_locs[i],
            draw_height,
            width=width,
            color=facecolor,
            edgecolor=edgecolor,
            linewidth=lw,
            hatch=hatch,
            zorder=3
        )

    ax.set_ylim(0, cap_pct)

    yticks = [5, 10, 15, 20, 25, 30]
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{int(y)}%"))

    ax.set_xticks(x_locs)
    ax.set_xticklabels(labels_sorted, rotation=90)

    ax.set_xlabel("Country / Tranche group", fontsize=12)
    ax.set_ylabel("Expected loss (%)", fontsize=12)

    ax.yaxis.grid(True)
    ax.xaxis.grid(False)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for i, actual in enumerate(values_sorted):
        x = x_locs[i]
        if actual > cap_pct:
            ax.annotate(
                "", xy=(x, cap_pct*0.985), xytext=(x, cap_pct*0.88),
                arrowprops=dict(arrowstyle='-|>', lw=1.0, color="black"),
                zorder=4
            )
            ax.text(x, cap_pct*0.995, f">{int(cap_pct)}%", ha="center", va="bottom", fontsize=9, zorder=4)
        else:
            ax.text(x, actual + 0.02*cap_pct, f"{int(round(actual))}%", ha="center", va="bottom", fontsize=9, zorder=4)

    legend_items = [
        Patch(facecolor=LIGHT_GRAY, edgecolor="none", label="EL by country (no tranching)"),
        Patch(facecolor=BAR_GRAY,   edgecolor="none", label="Tranches @ optimal sub (normalized)"),
        Patch(facecolor="black",    edgecolor="black", label="Tranches @ 30% sub (normalized)"),
        Patch(facecolor=POOL_FACE,  edgecolor=POOL_EDGE,  hatch=POOL_HATCH,  label="Pool EL"),
    ]
    leg = ax.legend(handles=legend_items, loc="upper left", frameon=True, fontsize=10,
                    facecolor="white", edgecolor="black")
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    return fig, ax

# ============================================================
# PLOT IMMEDIATELY
# ============================================================
fig, ax = plot_expected_loss_bars_tcopula(
    PDs=PDs,
    final_weights=final_weights,
    results_tranches=results_tranches_t,
    probabilities=probabilities,
    base_LGD=base_LGD,
    LGD_scalers=LGD_scalers,
    ref_date_str="2020-01",
    cap_pct=30.0
)
plt.savefig("FinalFigures/Expected_Loss_Bars_tcopula_2020-01.png", dpi=300, bbox_inches="tight")
plt.show()



#%%
# #Figure 11
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------ Load Germany weights (same as before) ------------
def _read_csv_forgiving(path, **kw):
    try:
        return pd.read_csv(path, **kw)
    except FileNotFoundError:
        return pd.read_csv(path.replace(" ", ""), **kw)

csv_path = "data /PDs+Weights.csv"
df = _read_csv_forgiving(csv_path, index_col=0)
df.index = pd.to_datetime(df.index, errors="coerce").to_period("M").to_timestamp("M")
df = df.sort_index()

w_cols = [c for c in df.columns if c.endswith("_Weights")]
final_weights = df[w_cols].astype(float).copy()
final_weights.columns = [c.replace("_Weights", "") for c in w_cols]
if "Germany" not in final_weights.columns:
    raise ValueError("Could not find Germany weights column (expected 'Germany_Weights').")

w_de = final_weights["Germany"].astype(float)
w_de = w_de.replace([np.inf, -np.inf], np.nan)
w_de = w_de.where(w_de > 0.0)

# ------------ Subordination levels ------------
subs = {
    "Sub = 15%":  0.15,
    "Sub = 30%":  0.30,
    "Sub = 45%":  0.45,
    "Sub = 61.6%":0.616,
    "Sub = 75%":  0.75,
}

# Average multiplier per sub:  m_t(s) = (1 - s) / w_DE,t
avg_mult = {}
for label, s in subs.items():
    mult_t = (1.0 - s) / w_de
    avg_mult[label] = float(np.nanmean(mult_t.values))

# ------------ Plot ------------
x_full = np.linspace(0.0, 6.0, 601)
x_trunc = np.linspace(0.0, 2.2, 200)
x_cutoff = 2.2
x_label_pos = 2.25

DARK_GREY  = "#555555"   # for vertical, optimal line, and 45° line
LIGHT_GREY = "#BFBFBF"   # for the other subordination lines

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif"],
    "mathtext.fontset": "cm",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 0.9,
})

fig, ax = plt.subplots(figsize=(5.0, 5.0))

# --- 45° diagonal line (dark grey dashed) ---
ax.plot(x_full, x_full, linestyle="--", linewidth=1.6, color=DARK_GREY)

# --- "45°" label near top right (slightly lower than line) ---
ax.text(
    0.97, 0.88, "45°",
    transform=ax.transAxes, ha="right", va="top", fontsize=11, color=DARK_GREY
)

# --- Vertical dashed line at x = 1.33 (dark grey) ---
ax.axvline(x=x_cutoff, color=DARK_GREY, linestyle="--", linewidth=1.6)

# --- Label ABOVE the x-axis (0.10 points above) and to the right ---
ax.text(
    x_cutoff + 0.05, 0.10, "x = 2.2",
    transform=ax.get_xaxis_transform(),
    ha="left", va="bottom", fontsize=10, color=DARK_GREY
)

# --- Subordination lines ---
for label, m in avg_mult.items():
    color = DARK_GREY if "61.6" in label else LIGHT_GREY
    y = m * x_trunc
    ax.plot(x_trunc, y, linestyle="-", linewidth=1.8, color=color)

    # Inline labels
    y_label = m * x_cutoff
    text = "Optimal subordination: 61.6%" if "61.6" in label else f"{label.replace('Sub = ', '')} subordination"
    ax.text(x_label_pos, y_label, text, fontsize=10, color=color, va="center", ha="left")

# --- Axes & styling ---
ax.set_xlim(0, 7)
ax.set_ylim(0, 7)
ax.set_xlabel("Safe assets used (€Tn)", fontsize=12)
ax.set_ylabel("Safe asset generated (€Tn)", fontsize=12)

# Integer ticks
ax.set_xticks(np.arange(0, 8, 1))
ax.set_yticks(np.arange(0, 8, 1))

# Grid & frame
ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.8)
ax.xaxis.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

# --- Save ---
os.makedirs("FinalFigures", exist_ok=True)
out_path = "FinalFigures/SafeAsset_Multiplier_Lines_45deg_Vertical_Above.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")
print("Saved:", out_path)





# %% 
# ---------- TABLE A: Averages (opt, fix, and delta) ----------
def _avg_tbl(metrics, cmp_df):
    rows = []
    for m in metrics:
        x = cmp_df.get(f"{m}_opt")
        y = cmp_df.get(f"{m}_fix")
        d = cmp_df.get(f"delta_{m}")
        if x is None or y is None or d is None:
            continue
        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(d)
        rows.append({
            "metric": m,
            "Avg(opt)": float(x[mask].mean()) if mask.any() else np.nan,
            "Avg(fix)": float(y[mask].mean()) if mask.any() else np.nan,
            "Avg delta (opt − fix)": float(d[mask].mean()) if mask.any() else np.nan,
        })
    return pd.DataFrame(rows).set_index("metric")

compare_means_tbl = _avg_tbl(metrics, cmp_df)
print("\n=== TABLE A — Averages (opt vs fixed) ===")
print(compare_means_tbl.round(6))
compare_means_tbl.to_csv("data /resutls analysis excel/t_copula_TABLE_comp.csv")

# ---------- TABLE B: Share of months with |delta| > threshold ----------
def _exceed_tbl(metrics, cmp_df, thresholds: dict, title: str):
    rows = []
    for m in metrics:
        d = cmp_df.get(f"abs_delta_{m}")
        thr = thresholds.get(m, np.nan)
        if d is None or not np.isfinite(thr):
            # keep row but mark N/A if no threshold
            rows.append({"metric": m, "threshold": thr, "share_|delta|>thr": np.nan, "N": int(0 if d is None else d.notna().sum())})
            continue
        arr = d.to_numpy(dtype=float)
        mask = np.isfinite(arr)
        share = float(np.mean(arr[mask] > thr)) if mask.any() else np.nan
        rows.append({"metric": m, "threshold": thr, "share_|delta|>thr": share, "N": int(mask.sum())})
    out = pd.DataFrame(rows).set_index("metric")
    print(f"\n=== TABLE B — {title} ===")
    print(out.round(6))
    return out

# (1) Your original (low) heuristics
thresholds_low = {
    'optimal_sub': 0.01,                  # 1% subordination
    'EL_pool': 2e-4,                      # 2 bps
    'EL_sen_optimal': 2e-4,               # 2 bps
    'EL_sen_30': 2e-4,                    # 2 bps
    'EL_jun_30': 2e-4,                    # 2 bps
    'EL_germany': 2e-4,                   # 2 bps
    'safe_asset_multiplier_optimal': 0.05,# as before
    'PoolLoss_p99': 0.01                  # absolute pool loss
}
exceed_tbl_low = _exceed_tbl(metrics, cmp_df, thresholds_low, "Share with |delta| above LOW heuristics")

# (2) HIGH heuristics you requested
#   optimal sub 2% (=0.02), EL pool 1% (=0.01), EL senior 1% (=0.01) [both definitions],
#   EL junior 1% (=0.01), EL Germany 0.5% (=0.005).
thresholds_high = thresholds_low.copy()
thresholds_high.update({
    'optimal_sub': 0.02,
    'EL_pool': 0.01,
    'EL_sen_optimal': 0.01,
    'EL_sen_30': 0.01,
    'EL_jun_30': 0.01,
    'EL_germany': 0.005,
    # keep these unchanged unless you want to raise them too:
    # 'safe_asset_multiplier_optimal': thresholds_high['safe_asset_multiplier_optimal'],
    # 'PoolLoss_p99': thresholds_high['PoolLoss_p99'],
})
exceed_tbl_high = _exceed_tbl(metrics, cmp_df, thresholds_high, "Share with |delta| above HIGH heuristics")

# (optional) save to CSV
# compare_means_tbl.to_csv("data /compare_means_opt_vs_fix.csv")
# exceed_tbl_low.to_csv("data /share_delta_exceed_low.csv")
# exceed_tbl_high.to_csv("data /share_delta_exceed_high.csv")

# %%
