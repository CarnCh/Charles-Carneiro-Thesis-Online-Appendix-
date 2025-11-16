#%% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as student_t, rankdata
from scipy.special import gammaln
import seaborn as sns

#%% 
path = "data /PDs+Weights.csv" 
df = pd.read_csv(path, index_col=0)
df.index = pd.to_datetime(df.index, errors="coerce").to_period("M").to_timestamp("M")
df = df.sort_index()

pd_cols = [c for c in df.columns if c.endswith("_PDs")]
w_cols  = [c for c in df.columns if c.endswith("_Weights")]
countries = sorted({c.replace("_PDs","") for c in pd_cols} & {c.replace("_Weights","") for c in w_cols})

PDs = df[[f"{c}_PDs" for c in countries]].astype(float).copy(); PDs.columns = countries  # LSEG 5y cumulative PDs (EOM)
final_weights = df[[f"{c}_Weights" for c in countries]].astype(float).copy(); final_weights.columns = countries

#%% Model settings 
probabilities = np.array([0.70, 0.25, 0.05])              # [good, mild, severe]
# Severity = np.array([0.01761628, 1., 6.8134313 ])         # from ESBies Dec-2015 mapping  <-- removed; computed via ln(x) inside split_pd_states
base_LGD = 0.60
LGD_scalers = np.array([0.75, 1.0, 1.25])
hurdle = 0.005
sub_grid = np.round(np.linspace(0.00, 0.90, 91), 3)
n = 500_000
rng = np.random.default_rng(7)
Germany = "Germany"
sub_fixed = 0.30

# Student-t copula params
nu_tcopula = 7.0         # degrees of freedom (grid-selected per window below)
win_months = 60          # rolling window length in months

# Fixed-lambda scenario for robustness (set to None to disable)
fixed_lambda_scenario = None   # e.g., 0.50 for a robustness run; None = not used here

# Adverse contagion rules (ESBies-style, mapped to High/Mid/Low buckets) 
# GE: all others +50%; FR: high 10%, mid/low 40%; IT & ES: high 5%, mid 10%, low 40%
# Buckets are derived from Moody's idealised 5y PD table 
adverse_rules = {
    "Germany": {"high": 0.50, "mid": 0.50, "low": 0.50},
    "France":  {"high": 0.10, "mid": 0.40, "low": 0.40},
    "Italy":   {"high": 0.05, "mid": 0.10, "low": 0.40},
    "Spain":   {"high": 0.05, "mid": 0.10, "low": 0.40},
}
core_triggers = ["Germany", "France", "Italy", "Spain"]

#%% Moody's idealised 5y PD → High/Mid/Low rating thresholds 
# High (Aaa–Aa) ≤ 0.12%; Mid (A–Baa) ≤ 1.64%; Low (Ba+) > 1.64%
def map_pd5y_to_bucket(pd5y: float) -> str:
    x = float(np.clip(pd5y, 0.0, 1.0))
    if x <= 0.0012:      # ≤ 0.12%
        return "high"
    elif x <= 0.0164:    # ≤ 1.64%
        return "mid"
    else:
        return "low"

#%% Helpers & function same as MC version 2, t-copula (without contagion)
def _nearest_psd(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    A = (A + A.T) / 2.0
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, eps, None)
    Apsd = (vecs * vals) @ vecs.T
    D = np.diag(1.0 / np.sqrt(np.clip(np.diag(Apsd), 1e-16, None)))
    C = D @ Apsd @ D
    return (C + C.T) / 2.0

# Diagnostics collector container 
tcopula_diag = []   # store per-month, per-ν rows

def _mean_offdiag(R: np.ndarray) -> float:
    d = R.shape[0]
    if d < 2: return 0.0
    s = (R.sum() - np.trace(R))
    return float(s / (d * (d - 1)))

# add EL_senior at fixed s-levels into the row dict 
def add_el_checkpoints(row_dict: dict, EL_senior_norm_curve: dict, levels=(0,10,20,30,40,50,60,70)):
    """
    Save **normalized-to-tranche** senior ELs at fixed s-levels.
    levels are in percentage points (ints). Keys: EL_sen_<pp> (e.g., EL_sen_30).
    Assumes your sub_grid contains these levels (step=0.01, 3 d.p.).
    """
    for pp in levels:
        s = round(pp / 100.0, 3)
        key = f"EL_sen_{pp:02d}"
        row_dict[key] = EL_senior_norm_curve.get(s, float('nan'))
    return row_dict

#t-copula pseudo-likelihood
def t_copula_loglik_from_z(z: np.ndarray, R: np.ndarray, nu: float) -> float:
    T, d = z.shape
    L = np.linalg.cholesky(R)
    logdetR = 2.0 * np.sum(np.log(np.diag(L)))
    Rinv = np.linalg.inv(R)

    const_d = gammaln((nu + d) / 2.0) - gammaln(nu / 2.0) - 0.5 * d * np.log(nu * np.pi)
    const_1 = gammaln((nu + 1) / 2.0) - gammaln(nu / 2.0) - 0.5 * np.log(nu * np.pi)

    Q = np.einsum('ti,ij,tj->t', z, Rinv, z)

    ll_mv = T * (const_d - 0.5 * logdetR) - ((nu + d) / 2.0) * np.sum(np.log1p(Q / nu))
    ll_uni = T * z.shape[1] * const_1 - ((nu + 1) / 2.0) * np.sum(np.log1p((z ** 2) / nu))
    return float(ll_mv - ll_uni)

def _maximize_lambda_likelihood(z: np.ndarray, S: np.ndarray, nu: float,
                                tol: float = 1e-4, max_iter: int = 60) -> tuple[float, float]:
    I = np.eye(S.shape[0])
    def f(lam: float) -> float:
        lam = float(np.clip(lam, 0.0, 1.0))
        R = lam * I + (1.0 - lam) * S
        try:
            return t_copula_loglik_from_z(z, R, nu)
        except np.linalg.LinAlgError:
            R_psd = _nearest_psd(R, eps=1e-10)
            return t_copula_loglik_from_z(z, R_psd, nu)

    f0 = f(0.0); f1 = f(1.0)
    a, b = 0.0, 1.0
    invphi = (np.sqrt(5) - 1) / 2.0
    invphi2 = (3 - np.sqrt(5)) / 2.0
    x1 = a + invphi2 * (b - a)
    x2 = a + invphi * (b - a)
    f1x = f(x1); f2x = f(x2)

    it = 0
    while (b - a) > tol and it < max_iter:
        if f1x < f2x:
            a = x1; x1 = x2; f1x = f2x
            x2 = a + invphi * (b - a)
            f2x = f(x2)
        else:
            b = x2; x2 = x1; f2x = f1x
            x1 = a + invphi2 * (b - a)
            f1x = f(x1)
        it += 1

    lam_star = x1 if f1x >= f2x else x2
    ll_star = max(f1x, f2x, f0, f1)
    if f0 >= ll_star and f0 >= f1:
        lam_star, ll_star = 0.0, f0
    elif f1 >= ll_star and f1 >= f0:
        lam_star, ll_star = 1.0, f1
    return float(np.clip(lam_star, 0.0, 1.0)), float(ll_star)

# Pseudo-observations with gentle de-tying noise
def window_pseudo_obs(hist_df: pd.DataFrame, seed: int = 0) -> np.ndarray:
    X = hist_df.select_dtypes(include="number").to_numpy(float)
    T, d = X.shape
    U = np.empty_like(X, dtype=float)
    for j in range(d):
        r = rankdata(X[:, j], method="average")
        U[:, j] = r / (T + 1.0)
    rng_local = np.random.default_rng(seed)
    eps = 1e-6 / (T + 2.0)
    U += (rng_local.random(U.shape) - 0.5) * eps
    return np.clip(U, 1e-12, 1 - 1e-12)

# Estimate correlation and nu (grid) per rolling window; optional fixed lambda
def estimate_corr_and_nu_grid(PDs_full: pd.DataFrame,
                              t_anchor: pd.Timestamp,
                              active_cols: list[str],
                              window_months: int = 60,
                              nu_grid=( 4, 5, 7, 10,15),
                              fixed_lambda: float | None = None,
                              diag_store: list | None = None
                              ) -> tuple[np.ndarray, float, float, list[str]]:
    
    start = (t_anchor.to_period("M") - window_months + 1).to_timestamp("M")
    hist_all = PDs_full.loc[(PDs_full.index >= start) & (PDs_full.index <= t_anchor), active_cols].sort_index()
    hist_all = hist_all.ffill().bfill()

    valid_mask = hist_all.notna().any(axis=0)
    used_cols = [c for c in hist_all.columns if valid_mask[c]]
    hist = hist_all[used_cols].copy()

    T = hist.shape[0]; d = hist.shape[1]
    if (d < 2) or (T < max(12, d + 2)):
        return np.eye(max(d, 1)), 7.0, 1.0, used_cols

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

        aic = 2 * 1 - 2 * ll  # k=1 (only ν counted; λ profiled)

        # ---- log diagnostics (one row per ν candidate) ----
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

# t-copula sampler
def sample_t_copula_uniforms(n_paths: int, corr: np.ndarray, nu: float, rng: np.random.Generator) -> np.ndarray:
    d = corr.shape[0]
    try: L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError: L = np.linalg.cholesky(_nearest_psd(corr, eps=1e-10))
    Z = rng.standard_normal((n_paths, d)) @ L.T
    S = rng.chisquare(df=nu, size=n_paths) / nu
    Tm = Z / np.sqrt(S)[:, None]
    return student_t.cdf(Tm, df=nu)

# State-splitting & LGD
def split_pd_states(pd5: np.ndarray, probabilities: np.ndarray, Severity: np.ndarray):
    # pd5 are 5y cumulative PDs; we scale them across macro states using ESBies Severity weights
    # >>> Updated: compute country-specific severity via ln-based formulas and ignore `Severity` argument.
    x = np.asarray(pd5, dtype=float)
    # decide units for ln(): if PDs look like fractions (<=1), convert to percent first
    with np.errstate(invalid='ignore'):
        median_pos = np.nanmedian(x[x > 0]) if np.any(x > 0) else 0.0
    x_pct = x * 100.0 if median_pos <= 1.0 else x  # use % in ln(.)
    eps = 1e-12
    ln_x = np.log(np.clip(x_pct, eps, None))
    mild   = 7.9265 * ln_x + 25.791
    severe = 11.419 * ln_x + 59.964
    Sev = np.vstack([np.ones_like(x), mild, severe])  # [3 x N]

    scale = (probabilities.reshape(-1, 1) * Sev).sum(axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    PDs_SxN = (Sev / scale) * x.reshape(1, -1)  # preserve unconditional PD level
    return np.clip(PDs_SxN, 0.0, 1.0)

def lgd_states_matrix(n_countries: int, base_LGD: float, LGD_scalers: np.ndarray):
    return (base_LGD * LGD_scalers)[:, None] * np.ones((1, n_countries))

def bund_el_from_pd(pd5_DE: float,
                    probabilities: np.ndarray, Severity: np.ndarray,
                    base_LGD: float, LGD_scalers: np.ndarray) -> float:
    PDs_states = split_pd_states(np.array([pd5_DE]), probabilities, Severity).flatten()
    LGD_states = base_LGD * LGD_scalers
    return float(np.sum(probabilities * PDs_states * LGD_states))

#%% New functions for Contagion
# Build bucket map (per month), categorize countries into High/Mid/Low buckets 
def build_bucket_map_from_pd5y(pd_row: pd.Series, used_cols: list[str]) -> dict:
    gmap = {}
    for c in used_cols:
        pd_c = float(pd_row.get(c, np.nan))
        gmap[c] = map_pd5y_to_bucket(pd_c)
    return gmap

#%% Apply adverse contagion bumps (conditional on trigger defaults)
# When designated countries defsault, it raises the PD of others
def apply_adverse_bumps(default_bool: np.ndarray,
                        used_cols: list[str],
                        triggers_idx: dict,
                        group_map: dict,
                        rng: np.random.Generator):
    """
    default_bool: (n_paths, N) after baseline default draws (0/1)
    triggers_idx: {'Germany': j_GE, 'France': j_FR, ...} subset of columns that exist
    group_map: {col -> 'high'|'mid'|'low'}
    Returns modified default_bool after applying bump draws.
    """
    n_paths, N = default_bool.shape
    present_triggers = {k: v for k, v in triggers_idx.items() if 0 <= v < N}
    if not present_triggers:
        return default_bool

    trig_paths = {k: default_bool[:, j].astype(bool) for k, j in present_triggers.items()}

    for j, c in enumerate(used_cols):
        if c in present_triggers.keys():
            continue  # bumps apply to "others"
        g = group_map.get(c, "low")
        one_minus = np.ones(n_paths, dtype=float)
        for k, active in trig_paths.items():
            if not np.any(active):
                continue
            p_k = adverse_rules[k][g]
            one_minus[active] *= (1.0 - p_k)
        p_comb = 1.0 - one_minus

        mask_candidates = (~default_bool[:, j]) & (p_comb > 0)
        if np.any(mask_candidates):
            u = rng.random(n_paths)
            bump = (u < p_comb) & mask_candidates
            default_bool[bump, j] = True

    return default_bool

#%% Core month simulation (ADVERSE contagion)
def simulate_month_tcopula_adverse(weights_row: pd.Series, pd5_row: pd.Series,
                                   probabilities: np.ndarray, Severity: np.ndarray,
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

    w0 = weights_row.fillna(0.0).copy()
    p0 = pd5_row.fillna(0.0).copy()
    active0 = list(w0.index[w0.values > 0])

    R, nu_hat, lam_hat, used_cols = estimate_corr_and_nu_grid(
        PDs_full, t_anchor, active0, window_months=window_months,
        nu_grid=( 4, 5, 7, 10,15), fixed_lambda=fixed_lambda, diag_store=diag_store
    )

    if len(used_cols) < 1:
        return np.nan, np.nan, {float(s): np.nan for s in sub_grid}, np.nan

    w = w0[used_cols].to_numpy(float)
    pd5 = np.clip(p0[used_cols].to_numpy(float), 0.0, 1.0)
    w_sum = w.sum()
    if w_sum <= 0:
        return np.nan, np.nan, {float(s): np.nan for s in sub_grid}, np.nan
    w /= w_sum

    S = len(probabilities); N = len(used_cols)
    PDs_SxN = split_pd_states(pd5, probabilities, Severity)
    LGD_SxN = lgd_states_matrix(N, base_LGD, LGD_scalers)

    if verbose:
        lam_tag = f"{lam_hat:.6f}" + (" (fixed)" if fixed_lambda is not None else "")
        print(f"[ADVERSE] Date={t_anchor:%Y-%m}, ν={nu_hat:>4.1f}, λ={lam_tag}, N={N}")

    # Buckets + triggers
    group_map = build_bucket_map_from_pd5y(pd5_row, used_cols)
    trig_idx = {k: used_cols.index(k) for k in core_triggers if k in used_cols}

    # Simulate states & defaults + contagion
    states = rng.choice(S, size=n, p=probabilities)
    losses = np.empty(n, dtype=float)
    for s in range(S):
        idx = np.where(states == s)[0]
        if idx.size == 0:
            continue
        U = sample_t_copula_uniforms(idx.size, R, nu=nu_hat, rng=rng)
        default = (U < PDs_SxN[s, :])                              # baseline
        default = apply_adverse_bumps(default.copy(), used_cols,   # contagion
                                      trig_idx, group_map, rng)
        losses[idx] = (default * (w * LGD_SxN[s, :])).sum(axis=1)

    # Pool EL (per pool notional)
    EL_pool = float(losses.mean())
    pool_loss_p99 = float(np.percentile(losses, 99))

    # --- Normalized senior EL curve ---
    EL_senior_norm = {}
    for s in sub_grid:
        s = float(s)
        ws = 1.0 - s
        el_sen_pool = float(np.mean(np.maximum(losses - s, 0.0)))   # per pool notional
        EL_senior_norm[s] = (el_sen_pool / ws) if ws > 0 else np.nan

    # s* from normalized senior hurdle
    s_star = np.nan
    for s in sub_grid:
        if EL_senior_norm[float(s)] <= hurdle:
            s_star = float(s); break

    return s_star, EL_pool, EL_senior_norm, pool_loss_p99

#%% Run series & build full output (ADVERSE only)
def run_and_build_adverse(PDs: pd.DataFrame, final_weights: pd.DataFrame,
                          probabilities, Severity, base_LGD, LGD_scalers,
                          n, sub_grid, rng, hurdle, Germany="Germany", s_fixed=0.30,
                          nu: float = 7.0, window_months: int = 60,
                          fixed_lambda: float | None = None, verbose=True,
                          diag_store: list | None = None):
    common_idx  = final_weights.index.intersection(PDs.index)
    common_cols = final_weights.columns.intersection(PDs.columns)
    W = final_weights.loc[common_idx, common_cols]
    P = PDs.loc[common_idx, common_cols]

    rows = []
    for t in W.index:
        s_star, EL_pool, EL_senior_norm_curve, pool_loss_p99 = simulate_month_tcopula_adverse(
            W.loc[t], P.loc[t],
            probabilities, Severity, base_LGD, LGD_scalers,
            n, sub_grid, rng, hurdle,
            nu=nu, PDs_full=PDs,
            window_months=window_months, t_anchor=t, verbose=verbose,
            fixed_lambda=fixed_lambda, diag_store=diag_store
        )

        # Bund EL (per-bond, own-notional)
        pd_de = float(P.loc[t].get(Germany, np.nan))
        EL_bund = np.nan if np.isnan(pd_de) else bund_el_from_pd(
            pd_de, probabilities, Severity, base_LGD, LGD_scalers
        )

        # @ s* (normalized to tranche notional)
        if np.isnan(s_star):
            EL_sen_opt = np.nan; EL_jun_opt = np.nan; senior_share_opt = np.nan
        else:
            EL_sen_opt = EL_senior_norm_curve[float(s_star)]
            EL_jun_opt = ((EL_pool - (1.0 - s_star) * EL_sen_opt) / s_star) if s_star > 0 else np.nan
            senior_share_opt = 1.0 - s_star

        # @ 30% (normalized)
        EL_sen_30 = EL_senior_norm_curve.get(float(s_fixed), np.nan)
        EL_jun_30 = ((EL_pool - (1.0 - s_fixed) * EL_sen_30) / s_fixed) if (np.isfinite(EL_sen_30) and s_fixed > 0) else np.nan
        senior_share_30 = 1.0 - s_fixed

        # Safe-asset multipliers unchanged
        w_de = float(W.loc[t].get(Germany, np.nan))
        if np.isnan(w_de) or w_de <= 0.0:
            safe_mult_opt = np.nan; safe_mult_30 = np.nan
        else:
            safe_mult_opt = senior_share_opt / w_de if not np.isnan(senior_share_opt) else np.nan
            safe_mult_30  = senior_share_30 / w_de

        row = {
            "Month": t,
            "optimal_sub": s_star,
            "EL_sen_optimal": EL_sen_opt,     # normalized
            "EL_jun_optimal": EL_jun_opt,     # normalized
            "EL_sen_30": EL_sen_30,           # normalized
            "EL_jun_30": EL_jun_30,           # normalized
            "EL_pool": EL_pool,               # per pool notional (context)
            "EL_germany": EL_bund,
            "safe_asset_multiplier_optimal": safe_mult_opt,
            "safe_asset_multiplier_30": safe_mult_30,
            "PoolLoss_p99": pool_loss_p99
        }
        row = add_el_checkpoints(row, EL_senior_norm_curve, levels=(0,10,20,30,40,50,60,70))
        rows.append(row)

    return pd.DataFrame(rows).set_index("Month").sort_index()

#%% Execute ADVERSE run 
print("\n=== ADVERSE CONTAGION ONLY (GE/FR/IT/ES conditional bumps) ===")
tcopula_diag.clear()
results_adverse = run_and_build_adverse(
    PDs, final_weights,
    probabilities, None, base_LGD, LGD_scalers,
    n, sub_grid, rng, hurdle, Germany=Germany, s_fixed=sub_fixed,
    nu=nu_tcopula, window_months=win_months, fixed_lambda=fixed_lambda_scenario, verbose=True,
    diag_store=tcopula_diag
)
diag_df = pd.DataFrame(tcopula_diag).sort_values(["Month","nu"]).reset_index(drop=True)
print(results_adverse.head(10))
results_adverse.to_csv("data /MC_results_tcopula_adverse.csv", float_format="%.6f")

#%% Convergence Test & Diagnostics (ADVERSE regime)
def convergence_test_tcopula_adverse(PDs, final_weights,
                                     probabilities, Severity, base_LGD, LGD_scalers,
                                     sub_grid, rng, hurdle, Germany="Germany", s_fixed=0.30,
                                     test_date="2015-12", n_values=None, repeats=10, base_seed=7,
                                     nu: float = 7.0, window_months: int = 60,
                                     fixed_lambda: float | None = None,
                                     diag_store: list | None = None):
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
    p_test = PDs.loc[test_timestamp, common_cols]

    rows = []
    tag = " (fixed λ)" if fixed_lambda is not None else ""
    print(f"Running convergence test (t-copula ADVERSE{tag})...")
    print(f"Test date: {test_timestamp}")
    print(f"Number of active countries: {(w_test.fillna(0) > 0).sum()}\n")

    for i, n_ in enumerate(n_values):
        print(f"Testing n={n_:,} ({i+1}/{len(n_values)})...")
        bucket = []
        for r in range(repeats):
            rng_local = np.random.default_rng(base_seed + r + 7919*n_)
            s_star, EL_pool, EL_senior_norm_curve, pool_loss_p99 = simulate_month_tcopula_adverse(
                w_test, p_test,
                probabilities, Severity, base_LGD, LGD_scalers,
                n_, sub_grid, rng_local, hurdle,
                nu=nu, PDs_full=PDs, window_months=window_months,
                t_anchor=test_timestamp, verbose=False, fixed_lambda=fixed_lambda, diag_store=diag_store
            )

            EL_sen_opt = EL_senior_norm_curve[float(s_star)] if not np.isnan(s_star) else np.nan
            EL_jun_opt = ((EL_pool - (1.0 - s_star) * EL_sen_opt) / s_star) if (not np.isnan(s_star) and s_star > 0) else np.nan
            EL_sen_30  = EL_senior_norm_curve.get(float(s_fixed), np.nan)
            EL_jun_30  = ((EL_pool - (1.0 - s_fixed) * EL_sen_30) / s_fixed) if (np.isfinite(EL_sen_30) and s_fixed > 0) else np.nan

            pd_de = float(p_test.get(Germany, np.nan))
            EL_bund = np.nan if np.isnan(pd_de) else bund_el_from_pd(
                pd_de, probabilities, Severity, base_LGD, LGD_scalers
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
            if arr.size == 0: return np.nan, np.nan
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
            'optimal_sub': mu_s,
            'EL_pool': mu_ep,
            'EL_sen_optimal': mu_es,   # normalized
            'EL_sen_30': mu_e30,       # normalized
            'EL_jun_30': mu_j30,       # normalized
            'EL_germany': mu_g,
            'PoolLoss_p99': mu_p99,
            'optimal_sub_se': se_s,
            'EL_pool_se': se_ep,
            'EL_sen_optimal_se': se_es,
            'EL_sen_30_se': se_e30,
            'EL_jun_30_se': se_j30,
            'EL_germany_se': se_g,
            'PoolLoss_p99_se': se_p99,
            'EL_pool_mc_error_pct': (se_ep / mu_ep * 100) if (mu_ep not in (0, np.nan)) else np.nan,
            'EL_sen_optimal_mc_error_pct': (se_es / mu_es * 100) if (mu_es not in (0, np.nan)) else np.nan,
            'PoolLoss_p99_mc_error_pct': (se_p99 / mu_p99 * 100) if (mu_p99 not in (0, np.nan)) else np.nan,
            'repeats': repeats
        })
    return pd.DataFrame(rows)


def plot_convergence(conv_results, metrics_to_plot=None):
    if metrics_to_plot is None:
        metrics_to_plot = ['EL_pool', 'optimal_sub', 'EL_sen_optimal', 'PoolLoss_p99']

    available_metrics = [m for m in conv_results.columns
                         if m in conv_results.columns and not conv_results[m].isna().all()]
    if not available_metrics:
        print("No valid metrics to plot!")
        return

    n_metrics = min(len(metrics_to_plot), 4)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, metric in enumerate(metrics_to_plot[:4]):
        if metric not in conv_results.columns: 
            axes[i].set_visible(False); continue
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
        }

    n_last = int(ns[-1])
    mu_last = float(mu[-1])
    ci_last = float(ci_half[-1])

    targets = []
    if abs_tol is not None:
        targets.append(abs_tol)
    if rel_tol is not None and mu_last != 0:
        targets.append(rel_tol * abs(mu_last))
    if not targets:
        return {"ok": False, "converged_at_n": None,
                "last_mu": mu_last, "last_ci_half": ci_last,
                "repeats": int(df['repeats'].iloc[-1]),
                "needed_n": None}
    target_half = min(targets)
    if target_half <= 0 or ci_last == 0:
        needed_n = n_last
    else:
        needed_n = int(np.ceil(n_last * (ci_last / target_half) ** 2))

    return {"ok": False, "converged_at_n": None,
            "last_mu": mu_last, "last_ci_half": ci_last,
            "repeats": int(df['repeats'].iloc[-1]),
            "needed_n": needed_n}

#%% Convergence Test (t-copula, ADVERSE)
print("="*50)
print("MONTE CARLO CONVERGENCE TEST — t-copula (ADVERSE contagion)")
print("="*50)
conv_results_adv = convergence_test_tcopula_adverse(
    PDs, final_weights,
    probabilities, None, base_LGD, LGD_scalers,
    sub_grid, rng, hurdle, Germany, sub_fixed,
    test_date="2015-12",
    n_values=[1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000, 300_000, 400_000, 500_000, 1_000_000],
    repeats=12, base_seed=7,
    nu=nu_tcopula, window_months=win_months, fixed_lambda=fixed_lambda_scenario
)
print("\nConvergence Results (means ± SE):")
print("-" * 80)
print(conv_results_adv.round(6))

print("\nGenerating convergence plots (ADVERSE)...")
plot_convergence(conv_results_adv, metrics_to_plot=['EL_pool', 'optimal_sub', 'EL_sen_optimal', 'PoolLoss_p99'])

print("\nStability Analysis (relative-change rule):")
print("-" * 40)
for metric, tol in [('EL_pool', 0.001), ('EL_sen_optimal', 0.001), ('PoolLoss_p99', 0.02)]:
    if metric in conv_results_adv.columns:
        stability = analyze_convergence_stability(conv_results_adv, metric, tolerance=tol)
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
    if metric in conv_results_adv.columns and f"{metric}_se" in conv_results_adv.columns:
        res = analyze_convergence_ci(conv_results_adv, metric=metric, conf=0.95,
                                     abs_tol=abs_tol, rel_tol=rel_tol)
        ci_decisions.append((metric, res))
        print(f"\n{metric}:")
        if res["ok"]:
            print(f"  Converged at n = {res['converged_at_n']:,}")
        else:
            need_txt = "unknown" if res.get("needed_n") is None else f"{res['needed_n']:,}"
            print("  Not yet converged by CI rule.")
            print(f"  Projected n to satisfy CI tolerances ≈ {need_txt}")
        print(f"  Last mean = {res.get('last_mu', np.nan):.6f}, last 95% half-CI = {res.get('last_ci_half', np.nan):.6f}, repeats = {res.get('repeats', np.nan)}")

# Optional: final CI-based recommendation across metrics
tested_max_n = int(conv_results_adv['n_simulations'].max())
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
    print(f"Based on the CI approach (ADVERSE), recommend n ≈ {recommended_n:,} simulations "
          f"(binding metric(s): {', '.join(binding_metrics)}).")
    print("="*70)
else:
    print("\nNo CI-based recommendation could be formed (missing SEs or metrics).")
#%%
import seaborn as sns
sns.set_context("talk")

# Ensure Month is monthly period for nicer x-axis grouping
diag_df["Month"] = pd.to_datetime(diag_df["Month"]).dt.to_period("M").dt.to_timestamp("M")

# (a) Chosen ν over time (AIC winner)
winner = diag_df.loc[diag_df.groupby("Month")["AIC"].idxmin()]
plt.figure(figsize=(11,4))
plt.step(winner["Month"], winner["nu"], where="mid", linewidth=2)
plt.title("Selected t-copula degrees of freedom (ν) over time (AIC-minimizing)")
plt.ylabel("ν (df)"); plt.xlabel("Month"); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()

""" # (b) AIC heatmap (Month × ν)
pivot = diag_df.pivot_table(index="Month", columns="nu", values="AIC", aggfunc="min")
# Optional: center by row-min to show ΔAIC structure clearly
pivot_delta = pivot.sub(pivot.min(axis=1), axis=0)
plt.figure(figsize=(11,6))
sns.heatmap(pivot_delta.T, cbar_kws={"label":"ΔAIC to row-min"}, square=False)
plt.title("ΔAIC by Month and ν (row-min subtracted)"); plt.xlabel("Month"); plt.ylabel("ν"); plt.tight_layout(); plt.show()
 """
# (c) Strength of selection: ΔAIC to 2nd best (per month)
def delta_to_runner_up(g):
    a = np.sort(g["AIC"].values)
    return a[1] - a[0] if len(a) >= 2 else np.nan
delta2 = diag_df.groupby("Month").apply(delta_to_runner_up).rename("deltaAIC_runnerup").reset_index()
plt.figure(figsize=(11,3.6))
plt.plot(delta2["Month"], delta2["deltaAIC_runnerup"], marker="o", lw=1.5)
plt.axhspan(0, 2, color="tab:gray", alpha=0.15, label="ΔAIC ≤ 2 (ties)")
plt.axhspan(4, 7, color="tab:orange", alpha=0.10, label="ΔAIC 4–7 (less support)")
plt.axhline(10, ls="--", alpha=0.6, label="ΔAIC = 10")
plt.title("Evidence for selected ν: ΔAIC to runner-up")
plt.ylabel("ΔAIC"); plt.xlabel("Month"); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

# (d) λ̂ path corresponding to the winning ν each month
plt.figure(figsize=(11,3.6))
plt.plot(winner["Month"], winner["lambda_hat"], lw=1.8)
plt.title("Estimated shrinkage λ̂ (for winning ν) over time")
plt.ylabel("λ̂"); plt.xlabel("Month"); plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
""" 
# (e) Optional: frequency of chosen ν
plt.figure(figsize=(6,3.8))
sns.countplot(data=winner, x="nu")
plt.title("Chosen ν (AIC) — frequency"); plt.xlabel("ν"); plt.ylabel("Months"); plt.tight_layout(); plt.show()
 """

#%% useful graphs and plot for comparison

#This re-runs the month simulation once but returns pathwise arrays so all tables/plots can be computed without re-simulating again.
def simulate_month_adverse_for_diagnostics(
    weights_row: pd.Series, pd5_row: pd.Series,
    probabilities: np.ndarray, Severity: np.ndarray,
    base_LGD: float, LGD_scalers: np.ndarray,
    n: int, rng: np.random.Generator,
    PDs_full: pd.DataFrame, window_months: int, t_anchor: pd.Timestamp,
    fixed_lambda: float | None = None,
    diag_store: list | None = None
):
    """
    Returns:
      dict with keys:
        'used_cols'            : list[str]
        'group_map'            : {country -> 'high'|'mid'|'low'}
        'triggers'             : list[str] present in used_cols (subset of core_triggers)
        'trigger_flags'        : (n, K) bool array; per path: which triggers defaulted post-bump
        'default_baseline'     : (n, N) bool array; defaults before bump
        'default_post'         : (n, N) bool array; defaults after bump
        'loss_baseline'        : (n,) float; losses with baseline defaults (before bump)
        'loss_post'            : (n,) float; losses after bump
        'states'               : (n,) ints in {0,1,2}
        'w'                    : (N,) country weights used (sum=1)
        'LGD_SxN'              : (S,N) LGD matrix
        'PDs_SxN'              : (S,N) state PDs matrix
    """
    # Estimate corr/nu and prep data (reusing your function)
    w0 = weights_row.fillna(0.0).copy()
    p0 = pd5_row.fillna(0.0).copy()
    active0 = list(w0.index[w0.values > 0])

    R, nu_hat, lam_hat, used_cols = estimate_corr_and_nu_grid(
        PDs_full, t_anchor, active0, window_months=window_months,
        nu_grid=( 4, 5, 7, 10,15), fixed_lambda=fixed_lambda, diag_store=diag_store
    )
    if len(used_cols) < 1:
        raise ValueError("No active countries for diagnostics at this date.")

    w = w0[used_cols].to_numpy(float)
    pd5 = p0[used_cols].to_numpy(float)
    w_sum = w.sum()
    if w_sum <= 0:
        raise ValueError("All active weights are zero for this date.")
    w /= w_sum
    S, N = len(probabilities), len(used_cols)

    PDs_SxN = split_pd_states(np.clip(pd5, 0, 1), probabilities, Severity)
    LGD_SxN = lgd_states_matrix(N, base_LGD, LGD_scalers)

    # Map buckets and trigger indices
    group_map = build_bucket_map_from_pd5y(pd5_row, used_cols)
    present_triggers = [k for k in core_triggers if k in used_cols]
    trig_idx = {k: used_cols.index(k) for k in present_triggers}

    # Draw states and copula uniforms
    states = rng.choice(S, size=n, p=probabilities)
    U = sample_t_copula_uniforms(n, R, nu=nu_hat, rng=rng)

    # Baseline defaults & losses (before adverse bumps)
    default_baseline = np.zeros((n, N), dtype=bool)
    for s in range(S):
        idx = np.where(states == s)[0]
        if idx.size:
            default_baseline[idx, :] = (U[idx, :] < PDs_SxN[s, :])
    loss_baseline = np.zeros(n, dtype=float)
    for s in range(S):
        idx = np.where(states == s)[0]
        if idx.size:
            loss_baseline[idx] = (default_baseline[idx] * (w * LGD_SxN[s, :])).sum(axis=1)

    # Apply bumps; detect bump-induced flips
    default_post = default_baseline.copy()
    default_post = apply_adverse_bumps(default_post, used_cols, trig_idx, group_map, rng)

    # Post-bump losses
    loss_post = np.zeros(n, dtype=float)
    for s in range(S):
        idx = np.where(states == s)[0]
        if idx.size:
            loss_post[idx] = (default_post[idx] * (w * LGD_SxN[s, :])).sum(axis=1)

    # Trigger flags *after* bumps (whether anchor itself defaulted in that path)
    K = len(present_triggers)
    trigger_flags = np.zeros((n, K), dtype=bool)
    for k, name in enumerate(present_triggers):
        j = trig_idx[name]
        trigger_flags[:, k] = default_post[:, j]

    return {
        'used_cols': used_cols,
        'group_map': group_map,
        'triggers': present_triggers,
        'trigger_flags': trigger_flags,
        'default_baseline': default_baseline,
        'default_post': default_post,
        'loss_baseline': loss_baseline,
        'loss_post': loss_post,
        'states': states,
        'w': w,
        'LGD_SxN': LGD_SxN,
        'PDs_SxN': PDs_SxN
    }

# Helpers for conditional expectations on a subset of paths
def el_senior_on_subset(losses: np.ndarray, sub_grid: np.ndarray, hurdle: float):
    """
    Compute **normalized** EL_senior(s) for all s and the smallest s hitting the hurdle.
    Returns: (EL_curve_norm_dict, optimal_s)
    """
    EL_curve_norm = {}
    for s in sub_grid:
        s = float(s)
        ws = 1.0 - s
        el_sen_pool = float(np.mean(np.maximum(losses - s, 0.0)))
        EL_curve_norm[s] = (el_sen_pool / ws) if ws > 0 else np.nan

    s_star = np.nan
    for s in sub_grid:
        if EL_curve_norm[float(s)] <= hurdle:
            s_star = float(s); break
    return EL_curve_norm, s_star


def cond_mask(trigger_flags: np.ndarray, k: int, want_trigger: bool):
    """Return boolean mask for paths where trigger k DID (want_trigger=True) or DID NOT default."""
    return trigger_flags[:, k] if want_trigger else ~trigger_flags[:, k]

#table 1: Trigger-conditional loss summary (per chosen month)
def make_trigger_conditional_summary_for_month(
    t: pd.Timestamp, s_fixed: float, sub_grid: np.ndarray,
    probabilities, Severity, base_LGD, LGD_scalers, hurdle: float,
    rng: np.random.Generator, n: int,
    PDs: pd.DataFrame, final_weights: pd.DataFrame,
    window_months: int = 60, fixed_lambda: float | None = None
) -> pd.DataFrame:
    W = final_weights.loc[t]; P = PDs.loc[t]
    diag = simulate_month_adverse_for_diagnostics(
        W, P, probabilities, Severity, base_LGD, LGD_scalers,
        n, rng, PDs, window_months, t, fixed_lambda=fixed_lambda, diag_store=None
    )
    losses = diag['loss_post']  # we evaluate summary "with contagion" for realism
    triggers = diag['triggers']; tf = diag['trigger_flags']
    rows = []
    for k, name in enumerate(triggers):
        m_trig = cond_mask(tf, k, True)
        m_not  = cond_mask(tf, k, False)
        p_trig = float(m_trig.mean())

        # Means of pool loss
        E_loss_trig = float(losses[m_trig].mean()) if m_trig.any() else np.nan
        E_loss_not  = float(losses[m_not].mean())  if m_not.any() else np.nan

        # EL_senior at fixed s
                # EL_senior at fixed s (normalized to senior notional)
        EL30_trig = (float(np.mean(np.maximum(losses[m_trig]-s_fixed, 0.0))) / (1.0 - s_fixed)) if m_trig.any() else np.nan
        EL30_not  = (float(np.mean(np.maximum(losses[m_not]-s_fixed, 0.0)))  / (1.0 - s_fixed)) if m_not.any()  else np.nan


        # Optimal subordination (search on sub_grid)
        ELcurve_trig, s_opt_trig = el_senior_on_subset(losses[m_trig], sub_grid, hurdle) if m_trig.any() else ({}, np.nan)
        ELcurve_not,  s_opt_not  = el_senior_on_subset(losses[m_not],  sub_grid, hurdle) if m_not.any()  else ({}, np.nan)

        # Tail (p99)
        p99_trig = float(np.percentile(losses[m_trig], 99)) if m_trig.any() else np.nan
        p99_not  = float(np.percentile(losses[m_not],  99)) if m_not.any()  else np.nan

        # % paths where bumps flipped at least one *non-trigger* name (diagnostic)
        flipped = (diag['default_post'] & ~diag['default_baseline'])
        # count flips on non-trigger columns
        non_trig_cols = [j for j,c in enumerate(diag['used_cols']) if c != name]
        bump_fired_share = float((flipped[:, non_trig_cols].any(axis=1) & m_trig).mean()) if m_trig.any() else np.nan

        rows.append({
            'Month': t, 'Trigger': name, 'P(trigger)': p_trig,
            'E[Loss|trigger]': E_loss_trig, 'E[Loss|¬trigger]': E_loss_not, 'Δ Loss': (E_loss_trig - E_loss_not) if (m_trig.any() and m_not.any()) else np.nan,
            'EL_senior(30%)|trigger': EL30_trig, 'EL_senior(30%)|¬trigger': EL30_not, 'Δ EL_senior(30%)': (EL30_trig - EL30_not) if (m_trig.any() and m_not.any()) else np.nan,
            'optimal_sub|trigger': s_opt_trig, 'optimal_sub|¬trigger': s_opt_not, 'Δ optimal_sub': (s_opt_trig - s_opt_not) if (np.isfinite(s_opt_trig) and np.isfinite(s_opt_not)) else np.nan,
            'PoolLoss_p99|trigger': p99_trig, 'PoolLoss_p99|¬trigger': p99_not, 'Δ p99': (p99_trig - p99_not) if (m_trig.any() and m_not.any()) else np.nan,
            '% paths bump-fired (given trigger)': bump_fired_share
        })
    out = pd.DataFrame(rows)
    return out[['Month','Trigger','P(trigger)',
                'E[Loss|trigger]','E[Loss|¬trigger]','Δ Loss',
                'EL_senior(30%)|trigger','EL_senior(30%)|¬trigger','Δ EL_senior(30%)',
                'optimal_sub|trigger','optimal_sub|¬trigger','Δ optimal_sub',
                'PoolLoss_p99|trigger','PoolLoss_p99|¬trigger','Δ p99',
                '% paths bump-fired (given trigger)']]

#%% table 2: Country × Trigger conditional default uplift (same month)


# === Academic-style conditional uplift matrix (0% white → 100% black) ===

def make_conditional_uplift_matrix_for_month(
    t: pd.Timestamp, rng: np.random.Generator, n: int,
    probabilities, Severity, base_LGD, LGD_scalers,
    PDs: pd.DataFrame, final_weights: pd.DataFrame,
    window_months: int = 60, fixed_lambda: float | None = None,
    clamp_nonnegative: bool = True, eps: float = 1e-12
) -> pd.DataFrame:
    """
    Compute the conditional uplift matrix for month `t`.

    Returns
    -------
    pd.DataFrame
        Rows = affected countries,
        Columns = trigger countries,
        Values = conditional uplift in DEFAULT PROBABILITY (fractions, e.g., 0.01 = 1pp),
                 defined as E[PD | trigger fires] - E[PD | trigger does NOT fire].
    """
    import numpy as np
    import pandas as pd

    # Extract data
    W = final_weights.loc[t]
    P = PDs.loc[t]

    diag = simulate_month_adverse_for_diagnostics(
        W, P, probabilities, Severity, base_LGD, LGD_scalers,
        n, rng, PDs, window_months, t, fixed_lambda=fixed_lambda, diag_store=None
    )

    posts    = diag['default_post']   # (n_paths, N_countries)
    cols     = diag['used_cols']
    tf       = diag['trigger_flags']  # (n_paths, K)
    triggers = diag['triggers']

    uplift = pd.DataFrame(0.0, index=cols, columns=triggers, dtype=float)

    # Compute conditional uplifts
    for k, name in enumerate(triggers):
        m_trig = cond_mask(tf, k, True)
        m_not  = cond_mask(tf, k, False)

        if (m_trig is None) or (m_not is None) or (not np.any(m_trig)) or (not np.any(m_not)):
            uplift[name] = np.nan
            continue

        diff = posts[m_trig].mean(axis=0) - posts[m_not].mean(axis=0)
        if clamp_nonnegative:
            diff = np.where(diff > eps, diff, 0.0)
        uplift[name] = diff

    return uplift


def plot_uplift_heatmap(uplift_df,
                        month_label: str | None = None,
                        name_to_code: dict | None = None,
                        annotate: bool = True,
                        figsize=(10.5, 6.5),
                        save_path: str | None = None):
    """
    Academic, black-and-white heatmap for the conditional uplift matrix (country × trigger).
    0% uplift = white, 100% (max) uplift = black.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.ticker import FuncFormatter
    from cycler import cycler

    # --- Style settings ---
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.prop_cycle": cycler("color", ["black"]),
    })

    if name_to_code is None:
        name_to_code = {
            "Germany":"DE","Netherlands":"NL","Luxembourg":"LU","Austria":"AT","Finland":"FI",
            "France":"FR","Belgium":"BE","Estonia":"EE","Slovakia":"SK","Ireland":"IE",
            "Latvia":"LV","Lithuania":"LT","Malta":"MT","Slovenia":"SI","Spain":"ES",
            "Italy":"IT","Portugal":"PT","Cyprus":"CY","Greece":"GR","Croatia":"HR"
        }

    # --- Labels ---
    rows = list(uplift_df.index)
    cols = list(uplift_df.columns)
    row_labels = [name_to_code.get(r, r) for r in rows]
    col_labels = [f"Contagion {name_to_code.get(c, c)}" for c in cols]

    # --- Data in percentage points ---
    M = uplift_df.to_numpy(dtype=float)
    if not np.isfinite(M).any():
        raise ValueError("Uplift matrix contains no finite values.")
    M_pct = M * 100.0

    # --- Normalize and colormap ---
    finite_vals = M_pct[np.isfinite(M_pct)]
    vmax = float(np.nanpercentile(finite_vals, 99))
    vmax = vmax if (np.isfinite(vmax) and vmax > 0) else 1.0
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("Greys")  # NOTE: not reversed → 0% white, max black

    # --- Plot ---
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(M_pct, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

    # Axes
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Trigger (Contagion Source)", fontsize=12)
    ax.set_ylabel("Affected Country", fontsize=12)

    # Title
    ttl = "Conditional default uplift (percentage points)"
    if month_label:
        ttl += f" — {month_label}"
    ax.set_title(ttl, fontsize=12)

    # Frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Uplift (%-points)", fontsize=11)
    cbar.outline.set_linewidth(0.8)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(round(y))}%"))

    # --- In-cell annotations ---
    if annotate:
        nrows, ncols = M_pct.shape
        for i in range(nrows):
            for j in range(ncols):
                val = M_pct[i, j]
                if np.isfinite(val):
                    txt = f"{int(round(val))}%"
                    # Use white text for high uplift cells (dark)
                    color = "white" if norm(val) > 0.6 else "black"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    return fig, ax


# ===== RUNNING PART =====
tbl2 = make_conditional_uplift_matrix_for_month(
    t_demo, rng=rng_demo, n=200_000,
    probabilities=probabilities, Severity=None, base_LGD=base_LGD, LGD_scalers=LGD_scalers,
    PDs=PDs, final_weights=final_weights, window_months=win_months, fixed_lambda=fixed_lambda_scenario,
    clamp_nonnegative=True
)
print("\nTABLE 2 — Conditional default uplift (country × trigger)\n", tbl2.round(6))
plot_uplift_heatmap(tbl2, annotate=True)



#%% table 3: Bucket-level impact by trigger (same month)
# Two flavors: (A) impact on post-bump default rates and loss contribution by bucket; (B) share of bump-induced defaults by bucket.
def make_bucket_level_impact_for_month(
    t: pd.Timestamp, rng: np.random.Generator, n: int,
    probabilities, Severity, base_LGD, LGD_scalers,
    PDs: pd.DataFrame, final_weights: pd.DataFrame,
    window_months: int = 60, fixed_lambda: float | None = None
) -> pd.DataFrame:
    W = final_weights.loc[t]; P = PDs.loc[t]
    diag = simulate_month_adverse_for_diagnostics(
        W, P, probabilities, Severity, base_LGD, LGD_scalers,
        n, rng, PDs, window_months, t, fixed_lambda=fixed_lambda, diag_store=None
    )
    cols  = diag['used_cols']
    group = diag['group_map']              # 'high'|'mid'|'low'
    tf    = diag['trigger_flags']
    triggers = diag['triggers']
    post   = diag['default_post']
    base   = diag['default_baseline']
    SxN_LGD = diag['LGD_SxN']
    w = diag['w']
    states = diag['states']

    # map to friendly rating names for reporting
    bucket2label = {'high': 'Aaa–Aa', 'mid': 'A–Baa', 'low': 'Ba+'}
    labels = {c: bucket2label.get(group[c], 'Ba+') for c in cols}

    rows = []
    for k, name in enumerate(triggers):
        m_trig = cond_mask(tf, k, True)
        m_not  = cond_mask(tf, k, False)
        if not m_trig.any() or not m_not.any():
            continue

        # Default rate delta by bucket (post-bump)
        for bucket_key, bucket_name in [('high','Aaa–Aa'), ('mid','A–Baa'), ('low','Ba+')]:
            bucket_cols = [j for j,c in enumerate(cols) if group[c] == bucket_key]
            if not bucket_cols:
                continue
            pd_trig = post[m_trig][:, bucket_cols].mean()   # scalar mean over paths and cols
            pd_not  = post[m_not][:, bucket_cols].mean()
            delta_pd = float(pd_trig - pd_not)

            # Δ contribution to pool loss (bps): use state LGD and w
            # compute expected loss contribution from bucket under trig vs not
            def exp_loss_subset_exact(mask):
                if not mask.any() or not bucket_cols:
                    return np.nan
                idx = np.where(mask)[0]
                # per-path, sum_j (default * w_j * LGD_state_j)
                accum = []
                for i in idx:
                    s = states[i]
                    contrib = (post[i, bucket_cols] * (w[bucket_cols] * SxN_LGD[s, bucket_cols])).sum()
                    accum.append(contrib)
                return float(np.mean(accum))

            EL_trig = exp_loss_subset_exact(m_trig)
            EL_not  = exp_loss_subset_exact(m_not)
            delta_EL = EL_trig - EL_not if (np.isfinite(EL_trig) and np.isfinite(EL_not)) else np.nan

            # Share of bump-induced defaults in bucket (only among trigger paths)
            bumped = (post & ~base)
            share_bumped_in_bucket = float(
                (bumped[m_trig][:, bucket_cols].any(axis=1)).mean()
            ) if m_trig.any() and bucket_cols else np.nan

            rows.append({
                'Month': t, 'Trigger': name, 'Bucket': bucket_name,
                'Δ default rate (post-bump)': delta_pd,
                'Δ loss contribution (bps of notional)': delta_EL,
                '% paths with any bump in bucket (|trigger)': share_bumped_in_bucket
            })
    return pd.DataFrame(rows)

#table 4: Rating mapping snapshot
def rating_name_from_bucket(pd5y: float) -> str:
    # Aaa–Aa ≤ 0.12%; A–Baa ≤ 1.64%; Ba+ > 1.64%
    x = float(np.clip(pd5y, 0.0, 1.0))
    if x <= 0.0012:      # 0.12%
        return "Aaa–Aa"
    elif x <= 0.0164:    # 1.64%
        return "A–Baa"
    else:
        return "Ba+"

def make_rating_snapshot_table(t: pd.Timestamp, PDs: pd.DataFrame) -> pd.DataFrame:
    row = PDs.loc[t].dropna().copy()
    df_map = (row.to_frame("PD_5y")
                 .assign(Rating=lambda d: d["PD_5y"].apply(rating_name_from_bucket))
                 .sort_values("PD_5y")
                 .reset_index()
                 .rename(columns={'index': 'Country'}))
    return df_map[['Country','PD_5y','Rating']]

# graph 1: Heatmap of conditional default uplift (country × trigger)
def plot_uplift_heatmap(uplift_df: pd.DataFrame, figsize=(9,6)):
    plt.figure(figsize=figsize)
    sns.heatmap(uplift_df, cmap='Reds', annot=False, fmt=".3f", cbar_kws={'label': 'Δ PD'})
    plt.title("Conditional Default Uplift: PD(target | trigger) − PD(target | ¬trigger)")
    plt.xlabel("Trigger"); plt.ylabel("Target Country")
    plt.tight_layout(); plt.show()

#%% graph 2: Waterfall of bump-attributed expected loss
#We attribute (loss_post − loss_baseline) to triggers on paths where those triggers defaulted. If multiple triggers default in the same path, we split the increment equally among them (simple, transparent rule)
def plot_bump_waterfall_for_month(
    t: pd.Timestamp, rng: np.random.Generator, n: int,
    probabilities, Severity, base_LGD, LGD_scalers,
    PDs: pd.DataFrame, final_weights: pd.DataFrame,
    window_months: int = 60, fixed_lambda: float | None = None
):
    """
    Waterfall of contagion bump attribution.
    - X labels: 'Contagion DE', 'Contagion FR', ...
    - Y axis: values multiplied by 100 and shown as percentages
    - Axis label simplified to 'Expected Loss (%)'
    - Bars have NO outlines (no edges)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    # --- Simulate diagnostics ---
    W = final_weights.loc[t]
    P = PDs.loc[t]
    diag = simulate_month_adverse_for_diagnostics(
        W, P, probabilities, Severity, base_LGD, LGD_scalers,
        n, rng, PDs, window_months, t, fixed_lambda=fixed_lambda, diag_store=None
    )
    base_loss = diag['loss_baseline']   # per path, fraction of notional
    post_loss = diag['loss_post']
    diff = post_loss - base_loss        # pathwise increments from contagion bump
    triggers = diag['triggers']
    tf = diag['trigger_flags']          # (n_paths, K)
    K = len(triggers)

    # --- Attribution computation (expected incremental EL per trigger) ---
    attributions = np.zeros(K, dtype=float)
    for i in range(len(diff)):
        if diff[i] <= 0:
            continue
        active = np.where(tf[i])[0]
        if active.size == 0:
            continue
        attributions[active] += diff[i] / active.size

    baseline_EL = float(np.mean(base_loss))     # fraction
    total_EL    = float(np.mean(post_loss))     # fraction
    incr_by_trig = attributions / len(diff)     # fraction per trigger, expected increment

    # --- Short country codes for x labels ---
    name_to_code = {
        "Germany": "DE", "France": "FR", "Italy": "IT", "Spain": "ES", "Netherlands": "NL",
        "Belgium": "BE", "Austria": "AT", "Finland": "FI", "Ireland": "IE", "Portugal": "PT",
        "Greece": "GR", "Luxembourg": "LU", "Slovenia": "SI", "Slovakia": "SK",
        "Lithuania": "LT", "Latvia": "LV", "Estonia": "EE", "Cyprus": "CY", "Malta": "MT",
        "Croatia": "HR"
    }

    # --- Build steps (convert to PERCENT values) ---
    labels = ["Baseline EL"]
    heights_pct = [baseline_EL * 100.0]  # baseline bar height in %

    for j, name in enumerate(triggers):
        code = name_to_code.get(name, name[:2].upper())
        labels.append(f"Contagion {code}")
        heights_pct.append(float(incr_by_trig[j] * 100.0))  # increment in %

    labels.append("Total EL")
    heights_pct.append(total_EL * 100.0)  # final bar is the total level (not an increment)

    # Compute bottoms for stacking the incremental bars only
    # baseline at x=0 → bottom=0
    # increments at x=1..K → bottom = cumulative baseline + previous increments
    cum_pct = [heights_pct[0]]  # cumulative after baseline
    for v in heights_pct[1:1+K]:
        cum_pct.append(cum_pct[-1] + v)
    # cum_pct has length K+1; we will use cum_pct[i-1] as bottom for increment i (i=1..K)

    # --- Plot style (journal B&W) ---
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "grid.color": "0.85",
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
    })

    # --- Draw figure (no outlines on bars) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_axisbelow(True)

    # Baseline bar (light gray), no edge
    ax.bar(0, heights_pct[0], width=0.6, color="#bfbfbf", edgecolor="none", linewidth=0)

    # Increment bars (mid gray), stacked on cumulative
    for i in range(1, 1 + K):
        ax.bar(i, heights_pct[i], bottom=cum_pct[i - 1], width=0.6,
               color="#7f7f7f", edgecolor="none", linewidth=0)

    # Total bar (black), no edge
    ax.bar(1 + K, heights_pct[-1], width=0.6, color="black", edgecolor="none", linewidth=0)

    # X axis labels
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha='right')

    # Y axis formatting → integers with % sign
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{int(round(y))}%"))
    ax.set_ylabel("Expected Loss (%)", fontsize=12)

    # Minimal framing & grid
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Title (subtle)
    ax.set_title(f"Waterfall Attribution of Contagion Bump — {t:%Y-%m}", fontsize=12)

    fig.tight_layout()
    plt.savefig("FinalFigures/bump_waterfall.png", dpi=300)
    plt.show()

#%%
# graph 3: Time-series ribbons of EL_senior(30%) | trigger vs ¬trigger
# This builds a panel for one trigger at a time across the full index.
# -*- coding: utf-8 -*-
# Academic, monochrome implementation of:
#   1) make_time_ribbon_EL30  (computes conditional EL@30%)
#   2) plot_time_ribbon       (journal-style B&W time ribbon plot)
#
# Notes:
# - Output ELs are fractions in the DF; the plot shows **percent** with integer tick labels.
# - Pure B&W styling: Computer Modern, thin ticks, horizontal dotted grid.
# - No color fills: the “ribbon” is rendered with a white face and a subtle diagonal hatch.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from cycler import cycler

# ============================================================
# COMPUTATION
# ============================================================
# --- drop-in replacement: adds compare_to param, default='post' keeps current behavior ---

def make_time_ribbon_EL30(
    trigger_name: str, dates: list[pd.Timestamp], s_fixed: float,
    probabilities, Severity, base_LGD, LGD_scalers, hurdle: float,
    rng_seed_base: int, n: int,
    PDs: pd.DataFrame, final_weights: pd.DataFrame,
    window_months: int = 60, fixed_lambda: float | None = None,
    compare_to: str = "baseline"   # 'post' (current), or 'baseline'
) -> pd.DataFrame:
    """
    compare_to:
      'post'     -> EL30|¬trigger uses post-bump losses conditioned on NOT firing (current ribbon)
      'baseline' -> EL30|¬trigger uses baseline (no-bump) losses unconditionally
    """
    rows = []
    for i, t in enumerate(dates):
        rng = np.random.default_rng(rng_seed_base + i*101)
        W = final_weights.loc[t]; P = PDs.loc[t]
        diag = simulate_month_adverse_for_diagnostics(
            W, P, probabilities, Severity, base_LGD, LGD_scalers,
            n, rng, PDs, window_months, t, fixed_lambda=fixed_lambda, diag_store=None
        )
        s = float(s_fixed)
        def EL_sen(L):
            return float(np.mean(np.maximum(L - s, 0.0)) / max(1.0 - s, 1e-12))

        if trigger_name not in diag['triggers']:
            rows.append({'Month': t, 'EL30|trigger': np.nan, 'EL30|¬trigger': np.nan, 'Δ': np.nan})
            continue

        k = diag['triggers'].index(trigger_name)
        m_trig = cond_mask(diag['trigger_flags'], k, True)
        m_not  = cond_mask(diag['trigger_flags'], k, False)

        L_post = diag['loss_post']
        L_base = diag['loss_baseline']

        # always compute the trigger leg on POST-bump, conditioned on firing
        el_trig = EL_sen(L_post[m_trig]) if m_trig.any() else np.nan

        # reference leg per requested comparator
        if compare_to == "baseline":
            el_not = EL_sen(L_base)  # unconditional baseline
        elif compare_to == "post":
            el_not = EL_sen(L_post[m_not]) if m_not.any() else np.nan
        else:
            raise ValueError("compare_to must be 'post' or 'baseline'.")

        rows.append({
            'Month': t,
            'EL30|trigger': el_trig,
            'EL30|¬trigger': el_not,
            'Δ': (el_trig - el_not) if (np.isfinite(el_trig) and np.isfinite(el_not)) else np.nan
        })

    return pd.DataFrame(rows).set_index('Month').sort_index()

# ============================================================
# PLOTTING (journal-style, monochrome)
# ============================================================
def plot_time_ribbon(df_el30: pd.DataFrame, trigger_name: str):
    """
    Publication-ready B&W plot of EL_senior(30%) over time, conditioned on a trigger.
    - Lines: solid black (trigger), dashed black (no trigger).
    - Ribbon: diagonal hatch (no gray fill), between the two series.
    - Y-axis: integer percents with '%' suffix.
    """
    # --- Global B&W style ---
    plt.rcParams.update({
        # Typography
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",

        # Axes
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.9,

        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,

        # Grid (horizontal dotted)
        "grid.color": "0.85",
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,

        # Monochrome
        "axes.prop_cycle": cycler("color", ["black"]),
    })

    x = df_el30.index
    # Convert to percent for plotting
    y1 = (df_el30['EL30|trigger'].values * 100.0).astype(float)
    y0 = (df_el30['EL30|¬trigger'].values * 100.0).astype(float)

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.set_axisbelow(True)

    # Lines
    ax.plot(x, y1, label=f'EL senior (30%) | {trigger_name} default', linewidth=1.6, linestyle='-')
    ax.plot(x, y0, label='EL senior (30%) | no trigger', linewidth=1.6, linestyle='--')

    # Y-axis formatting: integers with %
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, _: f"{int(round(y))}%"))
    ax.set_ylabel("Expected Loss (%)", fontsize=12)
    ax.set_xlabel("Year", fontsize=12)

    # Grid
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)

    # Minimal frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend: thin border, inside
    leg = ax.legend(loc="upper right", frameon=True, fontsize=10, facecolor="white", edgecolor="black")
    leg.get_frame().set_linewidth(0.8)

    # Subtle title (≤12 pt)
    ax.set_title(f"EL senior (30%) over time — conditioned on {trigger_name} default", fontsize=12)

    fig.tight_layout()
    plt.savefig(f"FinalFigures/time_ribbon_EL30_{trigger_name.replace(' ','_')}.png", dpi=300)
    plt.show()

#%% Running the grpahs and table
t_demo = pd.Timestamp("2015-12-31")
rng_demo = np.random.default_rng(7)

#%%
# Table 1
tbl1 = make_trigger_conditional_summary_for_month(
    t_demo, s_fixed=sub_fixed, sub_grid=sub_grid,
    probabilities=probabilities, Severity=None,
    base_LGD=base_LGD, LGD_scalers=LGD_scalers, hurdle=hurdle,
    rng=rng_demo, n=200_000,
    PDs=PDs, final_weights=final_weights,
    window_months=win_months, fixed_lambda=fixed_lambda_scenario
)
print("\nTABLE 1 — Trigger-conditional loss summary\n", tbl1.round(6))

#%% Table 2 + Heatmap (Graph 1)
tbl2 = make_conditional_uplift_matrix_for_month(
    t_demo, rng=rng_demo, n=200_000,
    probabilities=probabilities, Severity=None, base_LGD=base_LGD, LGD_scalers=LGD_scalers,
    PDs=PDs, final_weights=final_weights, window_months=win_months, fixed_lambda=fixed_lambda_scenario
)
print("\nTABLE 2 — Conditional default uplift (country × trigger)\n", tbl2.round(6))
plot_uplift_heatmap(tbl2)

#%% Table 3
tbl3 = make_bucket_level_impact_for_month(
    t_demo, rng=rng_demo, n=200_000,
    probabilities=probabilities, Severity=None, base_LGD=base_LGD, LGD_scalers=LGD_scalers,
    PDs=PDs, final_weights=final_weights, window_months=win_months, fixed_lambda=fixed_lambda_scenario
)
print("\nTABLE 3 — Bucket-level impact by trigger\n", tbl3.round(6))

# Table 4
snapshot = make_rating_snapshot_table(t_demo, PDs)
print(snapshot)
#%%
# Graph 2 — Waterfall
plot_bump_waterfall_for_month(
    t_demo, rng=rng_demo, n=500_000,
    probabilities=probabilities, Severity=None, base_LGD=base_LGD, LGD_scalers=LGD_scalers,
    PDs=PDs, final_weights=final_weights, window_months=win_months, fixed_lambda=fixed_lambda_scenario
)

#%%
# Graph 3 — Time ribbons (example for Spain)
dates_panel = list(results_adverse.index)
df_ribbon_SP = make_time_ribbon_EL30(
    "Spain", dates_panel, s_fixed=sub_fixed,
    probabilities=probabilities, Severity=None,
    base_LGD=base_LGD, LGD_scalers=LGD_scalers, hurdle=hurdle,
    rng_seed_base=77, n=100_000,
    PDs=PDs, final_weights=final_weights,
    window_months=win_months, fixed_lambda=fixed_lambda_scenario
)
plot_time_ribbon(df_ribbon_SP, "Spain")

# %%
