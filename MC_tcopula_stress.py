#%% mc_tcopula_adverse_contagion_dual.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t as student_t, rankdata
from scipy.special import gammaln
import seaborn as sns

#%% import data
path = "data /PDs+Weights.csv"
df = pd.read_csv(path, index_col=0)
df.index = pd.to_datetime(df.index, errors="coerce").to_period("M").to_timestamp("M")
df = df.sort_index()

pd_cols = [c for c in df.columns if c.endswith("_PDs")]
w_cols  = [c for c in df.columns if c.endswith("_Weights")]
countries = sorted({c.replace("_PDs","") for c in pd_cols} & {c.replace("_Weights","") for c in w_cols})

PDs = df[[f"{c}_PDs" for c in countries]].astype(float).copy(); PDs.columns = countries
final_weights = df[[f"{c}_Weights" for c in countries]].astype(float).copy(); final_weights.columns = countries

#%% BASE MODEL SETTINGS (carried from the baseline)
probabilities = np.array([0.70, 0.25, 0.05])              # [good, mild, severe]
# Severity = np.array([0.01761628, 1., 6.8134313 ])       # ESBies Dec-2015 mapping  <-- removed; computed via ln(.) inside split_pd_states
base_LGD = 0.60
LGD_scalers = np.array([0.75, 1.0, 1.25])
hurdle = 0.005 
sub_grid = np.round(np.linspace(0.00, 0.90, 91), 3)

# MC & general
n = 500_000
rng = np.random.default_rng(7)
Germany = "Germany"
sub_fixed = 0.30

# Rolling window for correlation/ν estimation
win_months = 60

# λ handling: profile (optimize) by default (set to None). Set to e.g. 0.50 if you ever want fixed-λ.
fixed_lambda_scenario = None

# CONTAGION REGIME SETTINGS
# ν_crisis fixed to 4.0 (heavy tails in doom-loop state)
NU_CRISIS_FIXED = 4.0

# Correlation spike multiplier in contagion (off-diagonals scaled then PSD-projected)
# (kept as an input knob; the adaptive κ will override it each window)
KAPPA_CORR = 1.50

# Rating-bucket crisis bumps (logit-space)
# Moody's idealised 5y PD → High/Mid/Low thresholds (cumulative PD) 
# High (Aaa–Aa) ≤ 0.12%; Mid (A–Baa) ≤ 1.64%; Low (Ba+) > 1.64%
def map_pd5y_to_bucket(pd5y: float) -> str:
    x = float(np.clip(pd5y, 0.0, 1.0))
    if x <= 0.0012:      # ≤ 0.12%
        return "high"
    elif x <= 0.0164:    # ≤ 1.64%
        return "mid"
    else:
        return "low"

# Default logit bumps by bucket (Odds Ratio≈1.35, 1.85, 2.21 → Δ≈0.30, 0.62, 0.79)
DEFAULT_LOGIT_DELTAS = {"high": 0.30, "mid": 0.62, "low": 0.79}
country_logit_overrides: dict[str, float] = {}  # e.g. {"Italy": 1.25}  (direct Δ in log-odds)

#%% Helpers
def _nearest_psd(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    A = (A + A.T) / 2.0
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, eps, None)
    Apsd = (vecs * vals) @ vecs.T
    D = np.diag(1.0 / np.sqrt(np.clip(np.diag(Apsd), 1e-16, None)))
    C = D @ Apsd @ D
    return (C + C.T) / 2.0

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

def offdiag_values(R: np.ndarray) -> np.ndarray:
    """Vector of off-diagonal correlations."""
    d = R.shape[0]
    mask = ~np.eye(d, dtype=bool)
    return R[mask]

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
    x1 = a + invphi2 * (b - a); x2 = a + invphi * (b - a)
    f1x = f(x1); f2x = f(x2)
    it = 0
    while (b - a) > tol and it < max_iter:
        if f1x < f2x:
            a = x1; x1 = x2; f1x = f2x
            x2 = a + invphi * (b - a); f2x = f(x2)
        else:
            b = x2; x2 = x1; f2x = f1x
            x1 = a + invphi2 * (b - a); f1x = f(x1)
        it += 1
    lam_star = x1 if f1x >= f2x else x2
    ll_star = max(f1x, f2x, f0, f1)
    if f0 >= ll_star and f0 >= f1: lam_star, ll_star = 0.0, f0
    elif f1 >= ll_star and f1 >= f0: lam_star, ll_star = 1.0, f1
    return float(np.clip(lam_star, 0.0, 1.0)), float(ll_star)

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
    U = np.clip(U, 1e-12, 1 - 1e-12)
    return U

# returns the average of all off-diagonal entries in a correlation matrix
def _mean_offdiag(R: np.ndarray) -> float:
    """Mean of off-diagonal correlations."""
    d = R.shape[0]
    if d < 2:
        return 0.0
    s = (R.sum() - np.trace(R))  # sum of off-diags
    return float(s / (d*(d-1)))

# Optimize kappa such that the mean off-diagonal correlation ≈ target_corr
# Computes a crisis-intensity multiplier that scales the base correlations so their average matches a target
def compute_adaptive_kappa(R_base: np.ndarray,
                           target_corr: float = 0.66,
                           allow_downscale: bool = False,
                           kappa_min: float = 1.0,
                           kappa_max: float = 2.0) -> float:
    """
    Choose κ so that mean off-diagonal correlation in crisis ≈ target_corr.
    If allow_downscale=False, κ is never < 1 (only 'spikes' correlations).
    """
    rho_bar = _mean_offdiag(R_base)
    eps = 1e-8

    if not allow_downscale and rho_bar >= target_corr - 1e-12:
        return 1.0

    if abs(rho_bar) < eps:
        kappa_raw = kappa_max if target_corr > 0 else kappa_min
    else:
        kappa_raw = target_corr / rho_bar

    if not allow_downscale:
        kappa_raw = max(1.0, kappa_raw)

    kappa = float(np.clip(kappa_raw, kappa_min, kappa_max))
    return kappa

# Scale off-diagonal correlations by an adaptively κ
def stress_corr_matrix_adaptive(R_base: np.ndarray,
                                target_corr: float = 0.66,
                                allow_downscale: bool = False,
                                kappa_min: float = 1.0,
                                kappa_max: float = 2.0,
                                psd_eps: float = 1e-10) -> tuple[np.ndarray, float]:
    """
    Scale off-diagonals by adaptive κ to target the desired mean correlation,
    then re-project to PSD and re-normalize the diagonal to 1.
    Returns (R_psd, kappa_used).
    """
    d = R_base.shape[0]
    if d < 2:
        return np.eye(d), 1.0

    kappa = compute_adaptive_kappa(R_base, target_corr, allow_downscale, kappa_min, kappa_max)

    R_crisis = R_base.copy()
    R_crisis *= kappa
    np.fill_diagonal(R_crisis, 1.0)

    R_psd = _nearest_psd(R_crisis, eps=psd_eps)
    np.fill_diagonal(R_psd, 1.0)
    return R_psd, kappa

def estimate_corr_and_nu_grid(PDs_full: pd.DataFrame,
                              t_anchor: pd.Timestamp,
                              active_cols: list[str],
                              window_months: int = 60,
                              nu_grid=(4, 5, 7, 10,15),
                              fixed_lambda: float | None = None
                              ) -> tuple[np.ndarray, float, float, list[str]]:
    """
    Returns (R_hat, nu_hat, lambda_used, used_cols) for the window ending at t_anchor.
    ν is selected by AIC across the provided grid; λ is profiled (unless fixed_lambda not None).
    """
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

        aic = 2 * 1 - 2 * ll  # k=1 (count ν only; λ profiled out)
        if (best is None) or (aic < best['aic']):
            best = {'R': R, 'nu': float(nu), 'lam': float(lam_hat), 'aic': float(aic)}

    return best['R'], best['nu'], best['lam'], used_cols

def sample_t_copula_uniforms(n_paths: int, corr: np.ndarray, nu: float, rng: np.random.Generator) -> np.ndarray:
    d = corr.shape[0]
    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(_nearest_psd(corr, eps=1e-10))
    Z = rng.standard_normal((n_paths, d)) @ L.T
    S = rng.chisquare(df=nu, size=n_paths) / nu
    Tm = Z / np.sqrt(S)[:, None]
    U = student_t.cdf(Tm, df=nu)
    return U

#%% ESBies slicing & Bund EL 
def split_pd_states(pd5: np.ndarray, probabilities: np.ndarray, Severity: np.ndarray):
    """
    Compute per-country macro-state PDs using ln-based severity multipliers:
      mild   = 7.9265 * ln(PD%) + 25.791
      severe = 11.419 * ln(PD%) + 59.964
    'good' severity is 1. We ignore the Severity argument (kept for compatibility).
    PD input can be in fractions (0–1). We convert to percent for ln(.).
    """
    x = np.asarray(pd5, dtype=float)
    # use percent in the logs if PDs look like fractions
    with np.errstate(invalid='ignore'):
        median_pos = np.nanmedian(x[x > 0]) if np.any(x > 0) else 0.0
    x_pct = x * 100.0 if median_pos <= 1.0 else x
    eps = 1e-12
    ln_x = np.log(np.clip(x_pct, eps, None))
    mild   = 7.9265 * ln_x + 25.791
    severe = 11.419 * ln_x + 59.964
    Sev = np.vstack([np.ones_like(x), mild, severe])  # shape (3, N)

    # preserve unconditional PD level across states (probability-weighted)
    scale = (probabilities.reshape(-1, 1) * Sev).sum(axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    PDs_SxN = (Sev / scale) * x.reshape(1, -1)
    return np.clip(PDs_SxN, 0.0, 1.0)

def lgd_states_matrix(n_countries: int, base_LGD: float, LGD_scalers: np.ndarray):
    return (base_LGD * LGD_scalers)[:, None] * np.ones((1, n_countries))

def bund_el_from_pd(pd5_DE: float,
                    probabilities: np.ndarray, Severity: np.ndarray,
                    base_LGD: float, LGD_scalers: np.ndarray) -> float:
    PDs_states = split_pd_states(np.array([pd5_DE]), probabilities, Severity).flatten()
    LGD_states = base_LGD * LGD_scalers
    return float(np.sum(probabilities * PDs_states * LGD_states))

#%% LOGIT-SPACE CRISIS PD BUMP (rating-bucket based)
# Probabilities to log-odds
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1-1e-12)
    return np.log(p/(1-p))

# Log-odds to probabilities 
def _inv_logit(x: np.ndarray) -> np.ndarray:
    return 1.0/(1.0 + np.exp(-x))

# Create a vector of logit deltas by rating bucket
def build_logit_deltas_by_bucket(used_cols: list[str], pd5: np.ndarray,
                                 deltas_by_bucket: dict[str, float] | None = None,
                                 overrides: dict[str, float] | None = None) -> np.ndarray:
    """
    Assign logit shocks by rating bucket (high/mid/low) from 5y PDs.
    """
    if deltas_by_bucket is None:
        deltas_by_bucket = DEFAULT_LOGIT_DELTAS
    overrides = overrides or {}
    out = np.empty(len(used_cols), dtype=float)
    for i, c in enumerate(used_cols):
        if c in overrides:
            out[i] = float(overrides[c])      # direct Δ in log-odds
        else:
            bucket = map_pd5y_to_bucket(pd5[i])
            out[i] = float(deltas_by_bucket[bucket])
    return out

# Apply additive shift in log-odds: logit(PD') = logit(PD) + Δ.
# PDs to stressed PDs, after being scaled by odds-ratio bumps
def stress_pd5_logit(pd5: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """
    Apply additive shift in log-odds: logit(PD') = logit(PD) + Δ.
    """
    return _inv_logit(_logit(pd5) + deltas)

#%% Month simulation with t-copula + contagion
# takes the stressed PDs and stressed correlation matrix as inputs
def simulate_month_tcopula_with_contagion(
    weights_row: pd.Series, pd5_row: pd.Series,
    probabilities: np.ndarray, Severity: np.ndarray,
    base_LGD: float, LGD_scalers: np.ndarray,
    n: int, sub_grid: np.ndarray,
    rng: np.random.Generator, hurdle: float,
    PDs_full: pd.DataFrame,
    window_months: int = 60,
    t_anchor: pd.Timestamp | None = None,
    verbose: bool = True,
    fixed_lambda: float | None = None,
    # contagion knobs
    pi_crisis: float = 0.10,
    kappa_corr: float = 1.50,
    nu_crisis: float = 4.0,
    deltas_by_bucket: dict[str, float] | None = None,
    country_overrides: dict[str, float] | None = None
):
    if t_anchor is None:
        t_anchor = getattr(weights_row, "name", PDs_full.index.max())

    w0 = weights_row.fillna(0.0).copy()
    p0 = pd5_row.fillna(0.0).copy()
    active0 = list(w0.index[w0.values > 0])

    R_base, nu_hat, lam_hat, used_cols = estimate_corr_and_nu_grid(
        PDs_full, t_anchor, active0, window_months=window_months,
        nu_grid=(4, 5, 7, 10, 15), fixed_lambda=fixed_lambda
    )
    nu_normal = float(nu_hat)

    if len(used_cols) < 1:
        return np.nan, np.nan, {float(s): np.nan for s in sub_grid}, np.nan, {}

    w = w0[used_cols].to_numpy(float)
    pd5 = np.clip(p0[used_cols].to_numpy(float), 0.0, 1.0)
    w_sum = w.sum()
    if w_sum <= 0:
        return np.nan, np.nan, {float(s): np.nan for s in sub_grid}, np.nan, {}
    w /= w_sum

    S, N = len(probabilities), len(used_cols)

    # Crisis-stressed PDs via logit bump
    delta_vec = build_logit_deltas_by_bucket(
        used_cols, pd5, deltas_by_bucket=deltas_by_bucket, overrides=country_overrides
    )
    pd5_crisis = stress_pd5_logit(pd5, delta_vec)

    # Crisis-stressed correlation (adaptive κ)
    R_crisis, kappa_used = stress_corr_matrix_adaptive(
        R_base, target_corr=0.66, allow_downscale=False, kappa_min=1.0, kappa_max=2.0
    )
    nu_c = float(nu_crisis)

    if verbose:
        lam_tag = f"{lam_hat:.6f}" + (" (fixed)" if fixed_lambda is not None else "")
        mode = "MIX" if 0.0 < pi_crisis < 1.0 else ("ALL-CRISIS" if pi_crisis >= 1.0 else "ALL-NORMAL")
        print(f"[{t_anchor:%Y-%m}] ν_normal={nu_normal:>4.1f}, ν_crisis={nu_c:>4.1f}, "
              f"λ={lam_tag}, N={N}, mode={mode}, π={pi_crisis:.0%}, κ_used={kappa_used:.3f}")
        print(f"  [adaptive-κ] base_meanρ={_mean_offdiag(R_base):.3f}  target={0.66:.2f}  crisis_meanρ≈{_mean_offdiag(R_crisis):.3f}")

    # Allocate paths to regimes
    if pi_crisis <= 0.0:
        idx_c = np.array([], dtype=int); idx_n = np.arange(n, dtype=int)
    elif pi_crisis >= 1.0:
        idx_c = np.arange(n, dtype=int); idx_n = np.array([], dtype=int)
    else:
        crisis_flags = rng.random(n) < float(pi_crisis)
        idx_c = np.where(crisis_flags)[0]; idx_n = np.where(~crisis_flags)[0]

    losses = np.empty(n, dtype=float)

    # NORMAL regime
    if idx_n.size:
        PDs_SxN_norm = split_pd_states(pd5, probabilities, Severity)
        LGD_SxN = lgd_states_matrix(N, base_LGD, LGD_scalers)
        s_norm = rng.choice(S, size=idx_n.size, p=probabilities)
        U_norm = sample_t_copula_uniforms(idx_n.size, R_base, nu=nu_normal, rng=rng)
        for s_state in range(S):
            jj = np.where(s_norm == s_state)[0]
            if jj.size == 0: continue
            jj_idx = idx_n[jj]
            default = (U_norm[jj] < PDs_SxN_norm[s_state, :])
            losses[jj_idx] = (default * (w * LGD_SxN[s_state, :])).sum(axis=1)

    # CRISIS regime
    if idx_c.size:
        PDs_SxN_cri = split_pd_states(pd5_crisis, probabilities, Severity)
        LGD_SxN = lgd_states_matrix(N, base_LGD, LGD_scalers)
        s_cri = rng.choice(S, size=idx_c.size, p=probabilities)
        U_cri = sample_t_copula_uniforms(idx_c.size, R_crisis, nu=nu_c, rng=rng)
        for s_state in range(S):
            jj = np.where(s_cri == s_state)[0]
            if jj.size == 0: continue
            jj_idx = idx_c[jj]
            default = (U_cri[jj] < PDs_SxN_cri[s_state, :])
            losses[jj_idx] = (default * (w * LGD_SxN[s_state, :])).sum(axis=1)

    # Pool metrics (per pool notional)
    EL_pool = float(np.mean(losses))
    pool_loss_p99 = float(np.percentile(losses, 99))

    # --- Normalized senior EL curve (per senior notional) ---
    EL_senior_norm = {}
    for s in sub_grid:
        s = float(s)
        ws = 1.0 - s
        el_sen_pool = float(np.mean(np.maximum(losses - s, 0.0)))   # per pool notional
        EL_senior_norm[s] = (el_sen_pool / ws) if ws > 0 else np.nan

    # Choose s* by normalized hurdle
    s_star = np.nan
    for s in sub_grid:
        if EL_senior_norm[float(s)] <= hurdle:
            s_star = float(s); break

    # extras (unchanged)
    base_off = offdiag_values(R_base)
    cri_off = offdiag_values(R_crisis)
    extras = {
        "kappa_used": float(kappa_used),
        "crisis_mean_rho": float(cri_off.mean()) if cri_off.size else np.nan,
        "base_mean_rho": float(base_off.mean()) if base_off.size else np.nan,
        "p90_offdiag_base": float(np.percentile(base_off, 90)) if base_off.size else np.nan,
        "p90_offdiag_crisis": float(np.percentile(cri_off, 90)) if cri_off.size else np.nan,
    }

    return s_star, EL_pool, EL_senior_norm, pool_loss_p99, extras


#%% Run over all months
def run_and_build_tcopula_with_contagion(
    PDs: pd.DataFrame, final_weights: pd.DataFrame,
    probabilities, Severity, base_LGD, LGD_scalers,
    n, sub_grid, rng, hurdle, Germany="Germany", s_fixed=0.30,
    window_months: int = 60, fixed_lambda: float | None = None, verbose=True,
    pi_crisis: float = 0.10, kappa_corr: float = 1.50,
    nu_crisis: float = 4.0,
    deltas_by_bucket: dict[str, float] | None = None, country_overrides: dict[str, float] | None = None
):
    common_idx  = final_weights.index.intersection(PDs.index)
    common_cols = final_weights.columns.intersection(PDs.columns)
    W = final_weights.loc[common_idx, common_cols]
    P = PDs.loc[common_idx, common_cols]

    rows = []
    for t in W.index:
        s_star, EL_pool, EL_senior_norm_curve, pool_loss_p99, extras = simulate_month_tcopula_with_contagion(
            W.loc[t], P.loc[t],
            probabilities, Severity, base_LGD, LGD_scalers,
            n, sub_grid, rng, hurdle,
            PDs_full=PDs, window_months=window_months, t_anchor=t, verbose=verbose,
            fixed_lambda=fixed_lambda,
            pi_crisis=pi_crisis, kappa_corr=kappa_corr, nu_crisis=nu_crisis,
            deltas_by_bucket=deltas_by_bucket, country_overrides=country_overrides
        )

        # Bund EL (per-bond, own-notional)
        pd_de = float(P.loc[t].get(Germany, np.nan))
        EL_bund = np.nan if np.isnan(pd_de) else bund_el_from_pd(
            pd_de, probabilities, Severity, base_LGD, LGD_scalers
        )

        # At s* (normalized to tranche notional)
        if np.isnan(s_star):
            EL_sen_opt = np.nan; EL_jun_opt = np.nan; senior_share_opt = np.nan
        else:
            EL_sen_opt = EL_senior_norm_curve[float(s_star)]
            EL_jun_opt = ((EL_pool - (1.0 - s_star) * EL_sen_opt) / s_star) if s_star > 0 else np.nan
            senior_share_opt = 1.0 - s_star

        # At fixed s = 0.30 (normalized)
        EL_sen_30 = EL_senior_norm_curve.get(float(s_fixed), np.nan)
        EL_jun_30 = ((EL_pool - (1.0 - s_fixed) * EL_sen_30) / s_fixed) if (np.isfinite(EL_sen_30) and s_fixed > 0) else np.nan
        senior_share_30 = 1.0 - s_fixed

        # Safe-asset multipliers unchanged
        w_de = float(W.loc[t].get(Germany, np.nan))
        if np.isnan(w_de) or w_de <= 0.0:
            safe_mult_opt = np.nan; safe_mult_30  = np.nan
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
            "PoolLoss_p99": pool_loss_p99,
            "kappa_used": extras.get("kappa_used", float("nan")),
            "crisis_mean_rho": extras.get("crisis_mean_rho", float("nan")),
            "base_mean_rho": extras.get("base_mean_rho", float("nan")),
            "p90_offdiag_base":   extras.get("p90_offdiag_base",   float("nan")),
            "p90_offdiag_crisis": extras.get("p90_offdiag_crisis", float("nan")),
        }

        row = add_el_checkpoints(row, EL_senior_norm_curve, levels=(0,10,20,30,40,50,60,70))
        rows.append(row)

    return pd.DataFrame(rows).set_index("Month").sort_index()

#%% Convergence Test
def convergence_test_contagion(PDs, final_weights,
                               probabilities, Severity, base_LGD, LGD_scalers,
                               sub_grid, rng, hurdle, Germany="Germany", s_fixed=0.30,
                               test_date="2015-12", n_values=None, repeats=10, base_seed=7,
                               window_months: int = 60,
                               fixed_lambda: float | None = None,
                               pi_crisis: float = 0.10, kappa_corr: float = 1.50,
                               nu_crisis: float = 4.0,
                               deltas_by_bucket: dict[str, float] | None = None,
                               country_overrides: dict[str, float] | None = None):
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
    regime_tag = f"mixed π={pi_crisis:.0%}" if 0.0 < pi_crisis < 1.0 else ("all-crisis" if pi_crisis >= 1.0 else "all-normal")
    print(f"Running convergence test (contagion, {regime_tag}{tag})...")
    print(f"Test date: {test_timestamp}")
    print(f"Number of active countries: {(w_test.fillna(0) > 0).sum()}\n")

    for i, n_ in enumerate(n_values):
        print(f"Testing n={n_:,} ({i+1}/{len(n_values)})...")
        bucket = []
        for r in range(repeats):
            rng_local = np.random.default_rng(base_seed + r + 7919*n_)
            s_star, EL_pool, EL_senior_norm_curve, pool_loss_p99, extras = simulate_month_tcopula_with_contagion(
                w_test, pd_test,
                probabilities, Severity, base_LGD, LGD_scalers,
                n_, sub_grid, rng_local, hurdle,
                PDs_full=PDs, window_months=window_months, t_anchor=test_timestamp, verbose=False,
                fixed_lambda=fixed_lambda,
                pi_crisis=pi_crisis, kappa_corr=kappa_corr, nu_crisis=nu_crisis,
                deltas_by_bucket=deltas_by_bucket, country_overrides=country_overrides
            )

            EL_sen_opt = EL_senior_norm_curve[float(s_star)] if not np.isnan(s_star) else np.nan
            EL_jun_opt = ((EL_pool - (1.0 - s_star) * EL_sen_opt) / s_star) if (not np.isnan(s_star) and s_star > 0) else np.nan
            EL_sen_30  = EL_senior_norm_curve.get(float(s_fixed), np.nan)
            EL_jun_30  = ((EL_pool - (1.0 - s_fixed) * EL_sen_30) / s_fixed) if (np.isfinite(EL_sen_30) and s_fixed > 0) else np.nan

            pd_de = float(pd_test.get(Germany, np.nan))
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

    # Project n if not converged: CI ∝ 1/√n  =>  n_needed = n_last * (ci_last/target)^2
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

#%% 1) MIXED REGIME: pi_crisis = 10% SKIP THIS?
print("\n=== ADVERSE CONTAGION — MIXED REGIME (pi_crisis = 10%) ===")
results_mixed = run_and_build_tcopula_with_contagion(
    PDs, final_weights,
    probabilities, None, base_LGD, LGD_scalers,
    n, sub_grid, rng, hurdle, Germany=Germany, s_fixed=sub_fixed,
    window_months=win_months, fixed_lambda=fixed_lambda_scenario, verbose=True,
    pi_crisis=0.10, kappa_corr=KAPPA_CORR, nu_crisis=NU_CRISIS_FIXED,
    deltas_by_bucket=DEFAULT_LOGIT_DELTAS, country_overrides=country_logit_overrides
)
print(results_mixed.head(10))
#results_mixed.to_csv("data /MC_results_tcopula_stress_10%.csv", float_format="%.6f")

#%% Convergence test (mixed regime) SKIP THIS?
print("="*50)
print("MONTE CARLO CONVERGENCE TEST — contagion (mixed regime = 10%)")
print("="*50)
conv_results_mixed = convergence_test_contagion(
    PDs, final_weights,
    probabilities, None, base_LGD, LGD_scalers,
    sub_grid, rng, hurdle, Germany, sub_fixed,
    test_date="2015-12",
    n_values=[1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000, 300_000, 400_000, 500_000, 1_000_000],
    repeats=12, base_seed=7,
    window_months=win_months, fixed_lambda=fixed_lambda_scenario,
    pi_crisis=0.10, kappa_corr=KAPPA_CORR, nu_crisis=NU_CRISIS_FIXED,
    deltas_by_bucket=DEFAULT_LOGIT_DELTAS, country_overrides=country_logit_overrides
)
print("\nConvergence Results (means ± SE):")
print("-" * 80)
print(conv_results_mixed.round(6))

print("\nGenerating convergence plots...")
plot_convergence(conv_results_mixed, metrics_to_plot=['EL_pool', 'optimal_sub', 'EL_sen_optimal', 'PoolLoss_p99'])

print("\nStability Analysis (relative-change rule):")
print("-" * 40)
for metric, tol in [('EL_pool', 0.001), ('EL_sen_optimal', 0.001), ('PoolLoss_p99', 0.02)]:
    if metric in conv_results_mixed.columns:
        stability = analyze_convergence_stability(conv_results_mixed, metric, tolerance=tol)
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
    if metric in conv_results_mixed.columns and f"{metric}_se" in conv_results_mixed.columns:
        res = analyze_convergence_ci(conv_results_mixed, metric=metric, conf=0.95,
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
tested_max_n = int(conv_results_mixed['n_simulations'].max())
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


#%% 2) ESBies-STYLE STRESS: all paths in doom loop (pi_crisis = 1.0)
print("\n=== ADVERSE CONTAGION — ESBies-STYLE STRESS (ALL PATHS CRISIS) ===")
results_all_crisis = run_and_build_tcopula_with_contagion(
    PDs, final_weights,
    probabilities, None, base_LGD, LGD_scalers,
    n, sub_grid, rng, hurdle, Germany=Germany, s_fixed=sub_fixed,
    window_months=win_months, fixed_lambda=fixed_lambda_scenario, verbose=True,
    pi_crisis=1.0, kappa_corr=KAPPA_CORR, nu_crisis=NU_CRISIS_FIXED,
    deltas_by_bucket=DEFAULT_LOGIT_DELTAS, country_overrides=country_logit_overrides
)
print(results_all_crisis.head(10))
#%%
results_all_crisis.to_csv("data /MC_results_tcopula_stress.csv", float_format="%.6f")

#%%  Convergence test (all-crisis regime)
print("="*50)
print("MONTE CARLO CONVERGENCE TEST — contagion (ESBies-style stress = all paths crisis)")
print("="*50)
conv_results_all_crisis = convergence_test_contagion(
    PDs, final_weights,
    probabilities, None, base_LGD, LGD_scalers,
    sub_grid, rng, hurdle, Germany, sub_fixed,
    test_date="2015-12",
    n_values=[1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000, 200_000, 300_000, 400_000, 500_000, 1_000_000],
    repeats=12, base_seed=7,
    window_months=win_months, fixed_lambda=fixed_lambda_scenario,
    pi_crisis=1.0, kappa_corr=KAPPA_CORR, nu_crisis=NU_CRISIS_FIXED,
    deltas_by_bucket=DEFAULT_LOGIT_DELTAS, country_overrides=country_logit_overrides
)
print("\nConvergence Results (means ± SE):")
print("-" * 80)
print(conv_results_all_crisis.round(6))

print("\nGenerating convergence plots...")
plot_convergence(conv_results_all_crisis, metrics_to_plot=['EL_pool', 'optimal_sub', 'EL_sen_optimal', 'PoolLoss_p99'])

print("\nStability Analysis (relative-change rule):")
print("-" * 40)
for metric, tol in [('EL_pool', 0.001), ('EL_sen_optimal', 0.001), ('PoolLoss_p99', 0.02)]:
    if metric in conv_results_all_crisis.columns:
        stability = analyze_convergence_stability(conv_results_all_crisis, metric, tolerance=tol)
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
    ('EL_sen_optimal', 0.00020, None),
    ('EL_pool',        0.00020, None),
    ('PoolLoss_p99',   None,    0.02)
]
ci_decisions2 = []
for metric, abs_tol, rel_tol in ci_checks:
    if metric in conv_results_all_crisis.columns and f"{metric}_se" in conv_results_all_crisis.columns:
        res = analyze_convergence_ci(conv_results_all_crisis, metric=metric, conf=0.95,
                                     abs_tol=abs_tol, rel_tol=rel_tol)
        ci_decisions2.append((metric, res))
        print(f"\n{metric}:")
        if res["ok"]:
            print(f"  Converged at n = {res['converged_at_n']:,}")
        else:
            need_txt = "unknown" if res["needed_n"] is None else f"{res['needed_n']:,}"
            print("  Not yet converged by CI rule.")
            print(f"  Projected n to satisfy CI tolerances ≈ {need_txt}")
        print(f"  Last mean = {res['last_mu']:.6f}, last 95% half-CI = {res['last_ci_half']:.6f}, repeats = {res['repeats']}")

tested_max_n2 = int(conv_results_all_crisis['n_simulations'].max())
per_metric_needed2 = []
binding_metrics2 = []
for metric, res in ci_decisions2:
    if res.get("ok", False):
        per_metric_needed2.append(int(res["converged_at_n"]))
    elif res.get("needed_n") is not None:
        per_metric_needed2.append(int(res["needed_n"]))
    else:
        per_metric_needed2.append(tested_max_n2)

if per_metric_needed2:
    recommended_n2 = max(per_metric_needed2)
    for metric, res in ci_decisions2:
        if (res.get('ok', False) and res.get('converged_at_n') == recommended_n2) or \
           (not res.get('ok', False) and res.get('needed_n') == recommended_n2):
            binding_metrics2.append(metric)
    print("\n" + "="*70)
    print(f"Based on the CI approach, I recommend n = {recommended_n2:,} simulations "
          f"(binding metric(s): {', '.join(binding_metrics2)}).")
    print("="*70)
else:
    print("\nNo CI-based recommendation could be formed (missing SEs or metrics).")

print("\nRecommendation:")
final_n = conv_results_mixed['n_simulations'].iloc[-1]
print(
    f"Based on this test, your current n = {n:,} appears adequate for the mean metrics if they converged by CI and relative rules."
    if final_n >= n else
    f"Potentially insufficient. Consider increasing to at least {final_n:,} or until CI-based criteria are met."
)

#%% Graphs and tables

# A) PD SNAPSHOT + BUCKET TRANSITION 
def rating_name_from_bucket(pd5y: float) -> str:
    x = float(np.clip(pd5y, 0.0, 1.0))
    if x <= 0.0012:      # 0.12%
        return "Aaa–Aa"
    elif x <= 0.0164:    # 1.64%
        return "A–Baa"
    else:
        return "Ba+"

def pd_snapshot_and_transition(t: pd.Timestamp,
                               PDs: pd.DataFrame,
                               final_weights: pd.DataFrame,
                               deltas_by_bucket: dict[str, float] | None = None,
                               country_overrides: dict[str, float] | None = None,
                               window_months: int = 60,
                               fixed_lambda: float | None = None):
    """
    Returns:
      snapshot_df: per-country table with PD_normal, PD_crisis, deltas, bucket labels
      trans_mat:   3x3 transition counts (rows=normal bucket, cols=crisis bucket)
    """
    # active universe (weight>0 at t)
    w_row = final_weights.loc[t].fillna(0.0).copy()
    active = list(w_row.index[w_row.values > 0])
    p_row = PDs.loc[t, active].astype(float).copy()
    pd5 = np.clip(p_row.values, 0.0, 1.0)

    # build deltas by bucket
    delta_vec = build_logit_deltas_by_bucket(active, pd5,
                                             deltas_by_bucket=deltas_by_bucket,
                                             overrides=country_overrides)
    pd5_crisis = stress_pd5_logit(pd5, delta_vec)

    df = (pd.DataFrame({"Country": active,
                        "PD_normal": pd5,
                        "PD_crisis": pd5_crisis})
            .assign(Delta=lambda d: d.PD_crisis - d.PD_normal,
                    PctDelta=lambda d: np.where(d.PD_normal>0,
                                                d.Delta/np.maximum(d.PD_normal,1e-12),
                                                np.nan),
                    Bucket_normal=lambda d: d.PD_normal.apply(rating_name_from_bucket),
                    Bucket_crisis=lambda d: d.PD_crisis.apply(rating_name_from_bucket))
            .sort_values("PD_normal")
            .reset_index(drop=True))

    # 3x3 transition matrix
    cats = ["Aaa–Aa","A–Baa","Ba+"]
    trans = pd.crosstab(pd.Categorical(df["Bucket_normal"], categories=cats),
                        pd.Categorical(df["Bucket_crisis"], categories=cats)).reindex(index=cats, columns=cats).fillna(0).astype(int)
    trans.index.name = "Normal →"
    trans.columns.name = "→ Crisis"

    return df, trans

#  B) CORRELATION SPIKE: SUMMARY + HEATMAPS + HISTOGRAMS 
def offdiag_values(R: np.ndarray) -> np.ndarray:
    """Vector of off-diagonal correlations."""
    d = R.shape[0]
    mask = ~np.eye(d, dtype=bool)
    return R[mask]

COUNTRY_TO_CODE = {
    "Germany":"DE","France":"FR","Italy":"IT","Spain":"ES","Netherlands":"NL","Belgium":"BE",
    "Austria":"AT","Finland":"FI","Ireland":"IE","Portugal":"PT","Greece":"GR","Luxembourg":"LU",
    "Cyprus":"CY","Malta":"MT","Slovenia":"SI","Slovakia":"SK","Lithuania":"LT","Latvia":"LV",
    "Estonia":"EE","Croatia":"HR","Bulgaria":"BG","Romania":"RO","Poland":"PL","Hungary":"HU"
}

def _labels_from_used_cols(used_cols: list[str]) -> list[str]:
    # Fallback: first 3 letters upper if not in dict
    return [COUNTRY_TO_CODE.get(c, c[:3].upper()) for c in used_cols]

#%%
# Change panel redesigned to MATCH the base/crisis panels:
# - Same lower-triangle layout
# - Same grayscale fill rule (darker = larger |value|)
# - Adaptive text color (white on dark cells)
# - Diagonal = WHITE fill with BLACK country code (no "Δ " prefix)
# - No hatches or extra symbols; only the title mentions ΔR

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

def _plot_corr_triangle_fill(R: np.ndarray,
                             codes: list[str],
                             title: str = "",
                             figsize=(7, 7),
                             decimals: int = 2,
                             dark_text_threshold: float = 0.65):
    d = R.shape[0]
    assert R.shape == (d, d) and len(codes) == d

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
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "grid.color": "0.85",
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
        "axes.prop_cycle": cycler("color", ["black"]),
    })

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.5, d - 0.5); ax.set_ylim(d - 0.5, -0.5)
    ax.set_aspect("equal"); ax.axis("off")

    off_abs = np.abs(R[~np.eye(d, dtype=bool)])
    vmax = float(np.nanmax(off_abs)) if off_abs.size else 1.0
    vmax = vmax if (np.isfinite(vmax) and vmax > 0) else 1.0

    edge_color = "0.85"; edge_lw = 0.6

    for i in range(d):
        for j in range(i + 1):
            if i == j:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     facecolor="white", edgecolor=edge_color, linewidth=edge_lw)
                ax.add_patch(rect)
                ax.text(j, i, codes[i], ha="center", va="center", fontsize=12, color="black")
            else:
                val = float(R[i, j])
                inten = float(np.clip(abs(val) / vmax, 0.0, 1.0))
                gray = 1.0 - inten
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     facecolor=(gray, gray, gray), edgecolor=edge_color, linewidth=edge_lw)
                ax.add_patch(rect)
                txt_color = "white" if inten > dark_text_threshold else "black"
                ax.text(j, i, f"{val:.{decimals}f}", ha="center", va="center", fontsize=11, color=txt_color)

    if title:
        ax.set_title(title, fontsize=12, pad=6)
    fig.tight_layout()
    return fig, ax

def _plot_corr_triangle_fill_diff(D: np.ndarray,
                                  codes: list[str],
                                  title: str = "",
                                  figsize=(7, 7),
                                  decimals: int = 2,
                                  dark_text_threshold: float = 0.65):
    """
    Difference panel with IDENTICAL styling to base/crisis:
    - Fill intensity by |Δρ|
    - Signed value printed in the cell WITHOUT a '+' for positives
    - Diagonal white with country codes (no prefix)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from cycler import cycler

    d = D.shape[0]
    assert D.shape == (d, d) and len(codes) == d

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
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "grid.color": "0.85",
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
        "axes.prop_cycle": cycler("color", ["black"]),
    })

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.5, d - 0.5); ax.set_ylim(d - 0.5, -0.5)
    ax.set_aspect("equal"); ax.axis("off")

    off_abs = np.abs(D[~np.eye(d, dtype=bool)])
    vmax = float(np.nanmax(off_abs)) if off_abs.size else 1.0
    vmax = vmax if (np.isfinite(vmax) and vmax > 0) else 1.0

    edge_color = "0.85"; edge_lw = 0.6

    for i in range(d):
        for j in range(i + 1):
            if i == j:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     facecolor="white", edgecolor=edge_color, linewidth=edge_lw)
                ax.add_patch(rect)
                ax.text(j, i, codes[i], ha="center", va="center", fontsize=12, color="black")
            else:
                val = float(D[i, j])              # signed Δρ
                inten = float(np.clip(abs(val) / vmax, 0.0, 1.0))
                gray = 1.0 - inten
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                     facecolor=(gray, gray, gray), edgecolor=edge_color, linewidth=edge_lw)
                ax.add_patch(rect)
                txt_color = "white" if inten > dark_text_threshold else "black"
                # NOTE: no leading '+' for positives; negatives show '-' naturally
                ax.text(j, i, f"{val:.{decimals}f}", ha="center", va="center", fontsize=11, color=txt_color)

    if title:
        ax.set_title(title, fontsize=12, pad=6)
    fig.tight_layout()
    return fig, ax


# --- Wrapper showing three panels with unified styling ---
def corr_spike_summary_and_plots(t: pd.Timestamp,
                                 PDs: pd.DataFrame,
                                 final_weights: pd.DataFrame,
                                 target_corr: float = 0.66,
                                 window_months: int = 60,
                                 fixed_lambda: float | None = None,
                                 figsize_heat=(19, 7),
                                 figsize_hist=(7, 4)):
    """
    Base | Crisis | ΔR (all lower-triangle, identical style).
    """
    w_row = final_weights.loc[t].fillna(0.0).copy()
    active = list(w_row.index[w_row.values > 0])
    R_base, nu_hat, lam_hat, used_cols = estimate_corr_and_nu_grid(
        PDs, t, active, window_months=window_months, nu_grid=(4,5,7,10,15),
        fixed_lambda=fixed_lambda
    )
    R_crisis, kappa_used = stress_corr_matrix_adaptive(
        R_base, target_corr=target_corr, allow_downscale=False,
        kappa_min=1.0, kappa_max=2.0
    )

    def offdiag_values(A):
        m = ~np.eye(A.shape[0], dtype=bool)
        return A[m].ravel()

    base_off = offdiag_values(R_base)
    cri_off  = offdiag_values(R_crisis)
    def pct(a,p): return float(np.percentile(a,p)) if a.size else np.nan

    ev_base = np.linalg.eigvalsh(R_base)[::-1]
    ev_cris = np.linalg.eigvalsh(R_crisis)[::-1]
    pc1_base = float(ev_base[0]/np.sum(ev_base)) if ev_base.size else np.nan
    pc1_cris = float(ev_cris[0]/np.sum(ev_cris)) if ev_cris.size else np.nan

    summary = pd.DataFrame({
        "metric": ["mean_offdiag","median_offdiag","p10_offdiag","p90_offdiag","PC1_variance_share","kappa_used"],
        "base":   [base_off.mean(), np.median(base_off), pct(base_off,10), pct(base_off,90), pc1_base, np.nan],
        "crisis": [cri_off.mean(),  np.median(cri_off),  pct(cri_off,10),  pct(cri_off,90),  pc1_cris, kappa_used]
    })
    print("\nCorrelation spike summary:\n", summary.round(4))

    codes = _labels_from_used_cols(used_cols) if '_labels_from_used_cols' in globals() else used_cols

    fig, axes = plt.subplots(1, 3, figsize=figsize_heat)
    plt.subplots_adjust(wspace=0.1)

    plt.sca(axes[0]); _plot_corr_triangle_fill(R_base,   codes, title="Base correlation (lower triangle)")
    plt.sca(axes[1]); _plot_corr_triangle_fill(R_crisis, codes, title="Crisis correlation (lower triangle)")
    D = R_crisis - R_base
    plt.sca(axes[2]); _plot_corr_triangle_fill_diff(D, codes, title="Change in correlation ΔR (lower triangle)")

    plt.show()

    # Histograms unchanged (monochrome outlines)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.linewidth": 0.9,
        "grid.color": "0.85",
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
    })
    cb, bb = np.histogram(base_off, bins=40, density=True)
    cc, bc = np.histogram(cri_off,  bins=40, density=True)
    cx_b = 0.5*(bb[:-1]+bb[1:])
    cx_c = 0.5*(bc[:-1]+bc[1:])
    plt.figure(figsize=figsize_hist)
    plt.plot(cx_b, cb, '-',  linewidth=1.4, color="black", label="base")
    plt.plot(cx_c, cc, '--', linewidth=1.4, color="black", label="crisis")
    plt.title(f"Off-diagonal correlations — {t:%Y-%m}", fontsize=12)
    plt.xlabel(r"$\rho_{ij}$  (i \ne j)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.grid(True, axis='y')
    leg = plt.legend(frameon=True, facecolor="white", edgecolor="black", fontsize=10)
    leg.get_frame().set_linewidth(0.8)
    plt.tight_layout(); plt.show()

    return summary, R_base, R_crisis, used_cols

#%%




# C) EL DECOMPOSITION: TABLE + WATERFALL 
def simulate_losses_given_inputs(
    w: np.ndarray, pd5_vec: np.ndarray,
    R: np.ndarray, nu: float,
    probabilities: np.ndarray, Severity: np.ndarray,
    base_LGD: float, LGD_scalers: np.ndarray,
    n: int, rng: np.random.Generator
) -> np.ndarray:
    """
    One-regime simulator: given PDs (5y), correlation R, and nu,
    draws ESBies macro states and t-copula uniforms, returns path losses.
    """
    S = len(probabilities); N = len(pd5_vec)
    PDs_SxN = split_pd_states(np.clip(pd5_vec,0,1), probabilities, Severity)
    LGD_SxN = lgd_states_matrix(N, base_LGD, LGD_scalers)

    states = rng.choice(S, size=n, p=probabilities)
    U = sample_t_copula_uniforms(n, R, nu=nu, rng=rng)

    losses = np.empty(n, dtype=float)
    for s in range(S):
        idx = np.where(states == s)[0]
        if idx.size == 0:
            continue
        default = (U[idx, :] < PDs_SxN[s, :])
        losses[idx] = (default * (w * LGD_SxN[s, :])).sum(axis=1)
    return losses

def el_decomposition_for_month(
    t: pd.Timestamp, PDs: pd.DataFrame, final_weights: pd.DataFrame,
    probabilities, Severity, base_LGD, LGD_scalers,
    nu_crisis: float = 4.0,
    target_corr: float = 0.66,
    n: int = 300_000, rng_seed: int = 123,
    window_months: int = 60, fixed_lambda: float | None = None,
    deltas_by_bucket: dict[str, float] | None = None,
    country_overrides: dict[str, float] | None = None,
    make_waterfall: bool = True
):
    """
    Returns: (table_df, details_dict) and optionally plots a waterfall.
    Scenarios:
      1) Base: PD=base,   R=R_base,   nu=nu_normal
      2) +Corr: PD=base,  R=R_crisis, nu=nu_normal
      3) +PD:   PD=crisis,R=R_base,   nu=nu_normal
      4) Full:  PD=crisis,R=R_crisis, nu=nu_crisis
    """
    # active set & inputs
    w_row = final_weights.loc[t].fillna(0.0).copy()
    active = list(w_row.index[w_row.values > 0])
    if len(active) < 1:
        raise ValueError("No active names at this date.")
    p_row = PDs.loc[t, active].astype(float).copy()
    w = w_row[active].to_numpy(float)
    w = w / w.sum()
    pd5 = np.clip(p_row.to_numpy(float), 0, 1)

    # rolling R_base & nu_normal
    R_base, nu_normal, lam_hat, used_cols = estimate_corr_and_nu_grid(
        PDs, t, active, window_months=window_months, nu_grid=(4,5,7,10,15),
        fixed_lambda=fixed_lambda
    )
    # align vectors to used_cols (should already match 'active', but just in case)
    if used_cols != active:
        idx_map = [active.index(c) for c in used_cols]
        w   = w[idx_map]
        pd5 = pd5[idx_map]

    # crisis PDs
    delta_vec = build_logit_deltas_by_bucket(used_cols, pd5,
                                             deltas_by_bucket=deltas_by_bucket,
                                             overrides=country_overrides)
    pd5_crisis = stress_pd5_logit(pd5, delta_vec)

    # crisis correlation
    R_crisis, kappa_used = stress_corr_matrix_adaptive(
        R_base, target_corr=target_corr, allow_downscale=False, kappa_min=1.0, kappa_max=2.0
    )

    rng = np.random.default_rng(rng_seed)

    # 1) Base
    L1 = simulate_losses_given_inputs(w, pd5, R_base, nu_normal,
                                      probabilities, Severity, base_LGD, LGD_scalers,
                                      n, rng)
    EL1 = float(L1.mean())

    # 2) +Corr
    rng2 = np.random.default_rng(rng_seed + 1)
    L2 = simulate_losses_given_inputs(w, pd5, R_crisis, nu_normal,
                                      probabilities, Severity, base_LGD, LGD_scalers,
                                      n, rng2)
    EL2 = float(L2.mean())

    # 3) +PD
    rng3 = np.random.default_rng(rng_seed + 2)
    L3 = simulate_losses_given_inputs(w, pd5_crisis, R_base, nu_normal,
                                      probabilities, Severity, base_LGD, LGD_scalers,
                                      n, rng3)
    EL3 = float(L3.mean())

    # 4) Full crisis (+ν & interactions)
    rng4 = np.random.default_rng(rng_seed + 3)
    L4 = simulate_losses_given_inputs(w, pd5_crisis, R_crisis, nu_crisis,
                                      probabilities, Severity, base_LGD, LGD_scalers,
                                      n, rng4)
    EL4 = float(L4.mean())

    # increments
    corr_eff = EL2 - EL1
    pd_eff   = EL3 - EL1
    nu_int   = EL4 - EL1 - corr_eff - pd_eff

    table = pd.DataFrame({
        "Scenario": ["Base","+Corr","+PD","+ν&interaction","Full crisis"],
        "EL_pool":  [EL1, EL2, EL3, EL1 + corr_eff + pd_eff + nu_int, EL4],
        "Increment vs Base": [0.0, corr_eff, pd_eff, nu_int, EL4 - EL1]
    })
    details = {
        "nu_normal": nu_normal, "nu_crisis": nu_crisis,
        "mean_offdiag_base": _mean_offdiag(R_base),
        "mean_offdiag_crisis": _mean_offdiag(R_crisis),
        "kappa_used": kappa_used
    }

    print("\nEL decomposition (bps of notional):\n", table.round(6))
    print("\nDetails:\n", pd.Series(details).round(4))

    if make_waterfall:
        labels = ["Base", "+Corr", "+PD", "+ν&int", "Full"]
        vals   = [EL1, corr_eff, pd_eff, nu_int, EL4]
        # build cumulative for plotting
        plt.figure(figsize=(8,4.5))
        running = EL1
        plt.bar([0], [EL1], width=0.6, label="Base")
        plt.bar([1], [corr_eff], bottom=[EL1], width=0.6, label="+Corr")
        plt.bar([2], [pd_eff], bottom=[EL1 + corr_eff], width=0.6, label="+PD")
        plt.bar([3], [nu_int], bottom=[EL1 + corr_eff + pd_eff], width=0.6, label="+ν&interaction")
        plt.bar([4], [EL4], width=0.6, label="Full", color="gray", alpha=0.3)
        plt.xticks(range(5), labels)
        plt.ylabel("Expected loss (fraction of notional)")
        plt.title(f"EL decomposition — {t:%Y-%m}")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout(); plt.show()

    return table, details

#%% Running tables and graphs

# A)
t_pick = pd.Timestamp("2022-03-31")
snap_df, trans_mat = pd_snapshot_and_transition(
    t_pick, PDs, final_weights,
     deltas_by_bucket=DEFAULT_LOGIT_DELTAS,
     country_overrides=country_logit_overrides,
     window_months=win_months)

print(snap_df.head(12))
print("\nTransition matrix (counts):\n", trans_mat)

#%%
# B) 
corr = corr_spike_summary_and_plots(
     pd.Timestamp("2018-03-31"), PDs, final_weights,
     target_corr=0.66, window_months=win_months)

#%%
# C)
t_pick = pd.Timestamp("2018-03-31")
decomp = el_decomposition_for_month(
     t_pick, PDs, final_weights,
     probabilities, None, base_LGD, LGD_scalers,
     nu_crisis=NU_CRISIS_FIXED, target_corr=0.66,
     n=300_000, rng_seed=123,
     window_months=win_months,
     deltas_by_bucket=DEFAULT_LOGIT_DELTAS, country_overrides=country_logit_overrides,
     make_waterfall=True)

print(decomp)

# %%
#--------------------------- FIGURE C — Correlation Structure Before vs. During Crisis (pi_crisis = 1.0)

print("\n=== FIGURE C: Correlation Structure Before vs. During Crisis ===")

# Pick representative date (e.g., December 2015)
t_corr = pd.Timestamp("2018-03-31")

# Estimate normal correlation matrix (R_base) and its parameters
R_base, nu_normal, lam_hat, used_cols = estimate_corr_and_nu_grid(
    PDs, t_corr, [c for c in final_weights.columns if final_weights.loc[t_corr, c] > 0],
    window_months=win_months, nu_grid=(4,5,7,10,15), fixed_lambda=fixed_lambda_scenario
)

# Build crisis correlation (adaptive κ to reach ρ ≈ 0.66)
R_crisis, kappa_used = stress_corr_matrix_adaptive(
    R_base, target_corr=0.66, allow_downscale=False, kappa_min=1.0, kappa_max=2.0
)

# --- Summaries ---
mean_r_base = _mean_offdiag(R_base)
mean_r_crisis = _mean_offdiag(R_crisis)
print(f"Mean off-diagonal correlation (base):   {mean_r_base:.3f}")
print(f"Mean off-diagonal correlation (crisis): {mean_r_crisis:.3f} (κ = {kappa_used:.2f})")

# --- Plot heatmaps side-by-side ---
codes = _labels_from_used_cols(used_cols)
vmin, vmax = min(R_base.min(), R_crisis.min()), max(R_base.max(), R_crisis.max())

fig, axes = plt.subplots(1, 2, figsize=(12,5))
sns.heatmap(R_base, vmin=vmin, vmax=vmax, cmap="coolwarm",
            cbar_kws={"label":"ρ"}, xticklabels=codes, yticklabels=codes, ax=axes[0])
axes[0].set_title("Normal regime (mean ρ ≈ {:.2f})".format(mean_r_base))
axes[0].tick_params(axis="x", labelrotation=45)

sns.heatmap(R_crisis, vmin=vmin, vmax=vmax, cmap="coolwarm",
            cbar_kws={"label":"ρ"}, xticklabels=codes, yticklabels=codes, ax=axes[1])
axes[1].set_title("Crisis regime (mean ρ ≈ {:.2f})".format(mean_r_crisis))
axes[1].tick_params(axis="x", labelrotation=45)

plt.suptitle("Correlation Structure Before vs. During Crisis — {}".format(t_corr.strftime("%b %Y")),
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.95])
plt.show()

# --- Optionally, show histogram of off-diagonal correlations ---
plt.figure(figsize=(7,4))
plt.hist(offdiag_values(R_base), bins=40, alpha=0.5, label="Normal", density=True)
plt.hist(offdiag_values(R_crisis), bins=40, alpha=0.5, label="Crisis", density=True)
plt.title(f"Distribution of Pairwise Correlations — {t_corr:%Y-%m}")
plt.xlabel("ρ_ij (i≠j)")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %%
