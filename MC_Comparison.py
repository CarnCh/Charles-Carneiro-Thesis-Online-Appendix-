#%% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns  

#%%
# Basic MC simulation
df_norm = pd.read_csv("data /MC_results.csv", index_col=0)
# Version 2 with t-copula, optimized lambda and fixed lambda (0.50)
df_copu = pd.read_csv("data /MC_results_tcopula.csv", index_col=0)
df_copu_fixedlambda = pd.read_csv("data /MC_results_tcopula_lambda050.csv", index_col=0)
# Version 3 with t-copula and adverse contagion
df_adv  = pd.read_csv("data /MC_results_tcopula_adverse.csv", index_col=0)
# Version 4 with t-copula and crisis regime (stress test)
df_stress  = pd.read_csv("data /MC_results_tcopula_stress.csv", index_col=0)

#%% Ensure datetime index (months) for plotting
for df in [df_norm, df_copu, df_copu_fixedlambda, df_adv, df_stress]:
    # Try to coerce to datetime if not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

#%% Rename columns with suffixes
df_norm   = df_norm.add_suffix("_MC")
df_copu   = df_copu.add_suffix("_Copula")
df_copu_fixedlambda = df_copu_fixedlambda.add_suffix("_Copula050")
df_adv    = df_adv.add_suffix("_AdvCon")
df_stress = df_stress.add_suffix("_Stress")

#%% Merge on Month index
df_all = pd.concat([df_norm, df_copu, df_copu_fixedlambda, df_adv, df_stress], axis=1)
#print(df_all.head(10))

#%% Compute averages
avg_table = df_all.mean(numeric_only=True)
print("\nAverages over all months:")
print(avg_table)

#%%
avg_df = pd.DataFrame(avg_table, columns=["Average"]).T
# avg_df.to_csv("data /MC_average_comparison.csv")

#%% Helper: date x-axis formatting (optional but nice)
def pretty_date_axis(ax):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

#%% ---------- Figure 1: Optimal Subordination s* ----------
plt.figure(figsize=(11,5.2))
plt.plot(df_all.index, df_all.get("optimal_sub_MC"),         label="MC", linewidth=1.6)
plt.plot(df_all.index, df_all.get("optimal_sub_Copula"),     label="Copula (opt λ)", linewidth=1.6)
plt.plot(df_all.index, df_all.get("optimal_sub_Copula050"),  label="Copula (λ=0.50)", linewidth=1.6)
plt.plot(df_all.index, df_all.get("optimal_sub_AdvCon"),     label="Copula + Contagion", linewidth=1.6)
plt.plot(df_all.index, df_all.get("optimal_sub_Stress"),     label="Crisis Regime (Stress)", linewidth=1.6)
plt.title("Optimal Subordination $s^*$ — Dependence, Contagion & Stress Raise Protection")
plt.ylabel("Subordination (fraction)"); plt.xlabel("Month")
plt.grid(True, alpha=0.3); plt.legend()
pretty_date_axis(plt.gca())
plt.tight_layout(); plt.show()

#%% ---------- Figure 2: Senior EL at s = 0.30 vs German Bund ----------
plt.figure(figsize=(11,5.2))
plt.plot(df_all.index, df_all.get("EL_sen_30_MC"),         label="MC", linewidth=1.6)
plt.plot(df_all.index, df_all.get("EL_sen_30_Copula"),     label="Copula (opt λ)", linewidth=1.6)
plt.plot(df_all.index, df_all.get("EL_sen_30_Copula050"),  label="Copula (λ=0.50)", linewidth=1.6)
plt.plot(df_all.index, df_all.get("EL_sen_30_AdvCon"),     label="Copula + Contagion", linewidth=1.6)
plt.plot(df_all.index, df_all.get("EL_sen_30_Stress"),     label="Crisis Regime (Stress)", linewidth=1.6)
# Bund: take from any (MC used here)
plt.plot(df_all.index, df_all.get("EL_germany_MC"), label="German Bund EL", linestyle="--", linewidth=1.6)
plt.title("Senior EL at Subordination = 0.30 vs German Bund — All Versions")
plt.ylabel("Expected Loss"); plt.xlabel("Month")
plt.grid(True, alpha=0.3); plt.legend()
pretty_date_axis(plt.gca())
plt.tight_layout(); plt.show()

#%% ---------- Figure 3: Safe-asset multipliers (optimal and s = 0.30) ----------
fig, ax = plt.subplots(2,1, figsize=(11,8), sharex=True)

# Optimal s*
ax[0].plot(df_all.index, df_all.get("safe_asset_multiplier_optimal_MC"),         label="MC", linewidth=1.6)
ax[0].plot(df_all.index, df_all.get("safe_asset_multiplier_optimal_Copula"),     label="Copula (opt λ)", linewidth=1.6)
ax[0].plot(df_all.index, df_all.get("safe_asset_multiplier_optimal_Copula050"),  label="Copula (λ=0.50)", linewidth=1.6)
ax[0].plot(df_all.index, df_all.get("safe_asset_multiplier_optimal_AdvCon"),     label="Copula + Contagion", linewidth=1.6)
ax[0].plot(df_all.index, df_all.get("safe_asset_multiplier_optimal_Stress"),     label="Crisis Regime (Stress)", linewidth=1.6)
ax[0].set_title("Safe-Asset Multiplier (Optimal $s^*$) — Output Safe Assets per € of Bund input")
ax[0].set_ylabel("Multiplier"); ax[0].grid(True, alpha=0.3); ax[0].legend()

# Fixed s = 0.30
ax[1].plot(df_all.index, df_all.get("safe_asset_multiplier_30_MC"),         label="MC", linewidth=1.6)
ax[1].plot(df_all.index, df_all.get("safe_asset_multiplier_30_Copula"),     label="Copula (opt λ)", linewidth=1.6)
ax[1].plot(df_all.index, df_all.get("safe_asset_multiplier_30_Copula050"),  label="Copula (λ=0.50)", linewidth=1.6)
ax[1].plot(df_all.index, df_all.get("safe_asset_multiplier_30_AdvCon"),     label="Copula + Contagion", linewidth=1.6)
ax[1].plot(df_all.index, df_all.get("safe_asset_multiplier_30_Stress"),     label="Crisis Regime (Stress)", linewidth=1.6)
ax[1].set_title("Safe-Asset Multiplier (Fixed s = 0.30)")
ax[1].set_ylabel("Multiplier"); ax[1].set_xlabel("Month"); ax[1].grid(True, alpha=0.3); ax[1].legend()

pretty_date_axis(ax[1])
plt.tight_layout(); plt.show()

#%% ---------- Figure 4: Scatter — EL_sen_30 comparisons with 45° line ----------
""" pairs = [
    ("EL_sen_30_MC",        "EL_sen_30_Copula",     "MC vs Copula (opt λ)"),
    ("EL_sen_30_MC",        "EL_sen_30_Copula050",  "MC vs Copula (λ=0.50)"),
    ("EL_sen_30_Copula",    "EL_sen_30_Copula050",  "Copula (opt λ) vs (λ=0.50)"),
    ("EL_sen_30_Copula050", "EL_sen_30_AdvCon",     "Copula (λ=0.50) vs Copula+Contagion"),
    ("EL_sen_30_AdvCon",    "EL_sen_30_Stress",     "Copula+Contagion vs Stress"),
    ("EL_sen_30_MC",        "EL_sen_30_AdvCon",     "MC vs Copula+Contagion"),
    ("EL_sen_30_MC",        "EL_sen_30_Stress",     "MC vs Stress"),
]
for x_col, y_col, ttl in pairs:
    if x_col in df_all.columns and y_col in df_all.columns:
        plt.figure(figsize=(6.4,6))
        x = df_all[x_col]; y = df_all[y_col]
        m = np.isfinite(x) & np.isfinite(y)
        plt.scatter(x[m], y[m], s=12, alpha=0.6)
        lo = np.nanmin([x[m].min(), y[m].min()]); hi = np.nanmax([x[m].max(), y[m].max()])
        plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, label="45° line")
        plt.title(f"Senior EL @ 0.30 — {ttl}")
        plt.xlabel(x_col.replace("_", " ")); plt.ylabel(y_col.replace("_", " "))
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()
 """
#%% ---------- Figure 5: Histograms — EL_jun_optimal (tail risk) ----------
plt.figure(figsize=(11,5.2))
hist_specs = [
    ("EL_jun_optimal_MC",        "MC"),
    ("EL_jun_optimal_Copula",    "Copula (opt λ)"),
    ("EL_jun_optimal_Copula050", "Copula (λ=0.50)"),
    ("EL_jun_optimal_AdvCon",    "Copula + Contagion"),
    ("EL_jun_optimal_Stress",    "Crisis Regime (Stress)"),
]
for col, lab in hist_specs:
    if col in df_all.columns:
        s = df_all[col].dropna()
        if len(s):
            plt.hist(s, bins=40, alpha=0.5, label=lab)
plt.title("Distribution of Junior EL at Optimal $s^*$ — All Versions")
plt.xlabel("EL_jun_optimal"); plt.ylabel("Frequency")
plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()

#%% ---------- Figure 6: Rolling 12M volatility — EL_sen_30 ----------
roll = 12
vol = pd.DataFrame({
    "MC":         df_all.get("EL_sen_30_MC"),
    "Copula":     df_all.get("EL_sen_30_Copula"),
    "Copula050":  df_all.get("EL_sen_30_Copula050"),
    "AdvCon":     df_all.get("EL_sen_30_AdvCon"),
    "Stress":     df_all.get("EL_sen_30_Stress"),
}).rolling(roll).std()
plt.figure(figsize=(11,5.2))
for c, lab in [("MC","MC"), ("Copula","Copula (opt λ)"), ("Copula050","Copula (λ=0.50)"),
               ("AdvCon","Copula + Contagion"), ("Stress","Crisis Regime (Stress)")]:
    if c in vol.columns:
        plt.plot(vol.index, vol[c], label=lab, linewidth=1.6)
plt.title(f"Rolling {roll}-Month Volatility of Senior EL @ 0.30 — All Versions")
plt.ylabel("Rolling Std. Dev."); plt.xlabel("Month")
plt.grid(True, alpha=0.3); plt.legend()
pretty_date_axis(plt.gca())
plt.tight_layout(); plt.show()

#%% ---------- Figure 7: Pool99 comparison ----------
plt.figure(figsize=(11,5.2))
plt.plot(df_all.index, df_all.get("PoolLoss_p99_MC"),        label="MC", linewidth=1.6)
plt.plot(df_all.index, df_all.get("PoolLoss_p99_Copula"),    label="Copula (opt λ)", linewidth=1.6)
plt.plot(df_all.index, df_all.get("PoolLoss_p99_Copula050"), label="Copula (λ=0.50)", linewidth=1.6)
plt.plot(df_all.index, df_all.get("PoolLoss_p99_AdvCon"),    label="Copula + Contagion", linewidth=1.6)
plt.plot(df_all.index, df_all.get("PoolLoss_p99_Stress"),    label="Crisis Regime (Stress)", linewidth=1.6)
plt.title("Pool99 Comparison — All Monte Carlo Versions")
plt.ylabel("Pool99"); plt.xlabel("Month")
plt.grid(True, alpha=0.3); plt.legend()
pretty_date_axis(plt.gca())
plt.tight_layout(); plt.show()


#%% ---------------------------------------------------------------
# EXTRA COMPARISON PLOTS 
#---------------------------------------------------------------
sns.set_context("talk", rc={"lines.linewidth":1.6})
sns.set_style("whitegrid")

# Helper: available versions & a safe getter
VERSIONS = {
    "MC": "MC",
    "Copula (opt λ)": "Copula",
    "Copula (λ=0.50)": "Copula050",
    "Copula+Contagion": "AdvCon",
    "Stress": "Stress",
}

def col(name, suffix):
    """Return df_all column if present, else None."""
    c = f"{name}_{suffix}"
    return c if c in df_all.columns else None

# Build a long-form helper for a chosen metric (e.g. EL_sen_30)
def build_long(metric_name):
    frames = []
    for label, suf in VERSIONS.items():
        c = col(metric_name, suf)
        if c:
            tmp = df_all[[c]].rename(columns={c: "value"}).copy()
            tmp["Version"] = label
            tmp["Month"] = tmp.index
            frames.append(tmp)
    return pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()

#===============================================================
# 1) BOX & VIOLIN PLOTS — distributional comparison (levels)
#===============================================================
for metric in ["EL_sen_30", "EL_jun_optimal", "PoolLoss_p99"]:
    long_df = build_long(metric)
    if long_df.empty: 
        continue
    plt.figure(figsize=(11,5.8))
    sns.boxplot(data=long_df, x="Version", y="value")
    plt.title(f"{metric} — Boxplot by Version"); plt.xlabel(""); plt.ylabel(metric)
    plt.tight_layout(); plt.show()

#===============================================================
# 2) ECDFs — tail comparison at a glance (Senior EL @ 0.30)
#===============================================================
long_sen = build_long("EL_sen_30")
if not long_sen.empty:
    plt.figure(figsize=(11,5.8))
    sns.ecdfplot(data=long_sen, x="value", hue="Version")
    plt.title("ECDF — Senior EL at s=0.30 (All Versions)")
    plt.xlabel("EL_sen_30"); plt.ylabel("F(x)")
    plt.tight_layout(); plt.show()


#===============================================================
# 4) CORRELATION HEATMAP — EL_sen_optima cross-version
#===============================================================
corr_cols = [col("EL_sen_optimal", suf) for suf in VERSIONS.values()]
corr_cols = [c for c in corr_cols if c]
if len(corr_cols) >= 2:
    C = df_all[corr_cols].corr()
    # friendlier labels
    C.index = [k for k in VERSIONS.keys() if col("EL_sen_optimal", VERSIONS[k])]
    C.columns = C.index
    plt.figure(figsize=(7.2,6.4))
    sns.heatmap(C, annot=True, fmt=".2f", square=True, cbar=True)
    plt.title("Correlation Heatmap — EL_sen_optimal across Versions")
    plt.tight_layout(); plt.show()

#===============================================================
# 5) ROLLING CORRELATION vs MC (EL_sen_30)
#===============================================================
base_col = col("EL_sen_30","MC")
if base_col:
    for label, suf in VERSIONS.items():
        c = col("EL_sen_30", suf)
       
base = df_all[base_col] if base_col else None
if base is not None:
    win = 12
    plt.figure(figsize=(11,5.2))
    for label, suf in VERSIONS.items():
        c = col("EL_sen_30", suf)
        if c and label != "MC":
            rc = base.rolling(win).corr(df_all[c])
            plt.plot(rc.index, rc, label=label)
    plt.title(f"Rolling {win}-Month Correlation vs MC — EL_sen_30")
    plt.ylabel("Rolling corr"); plt.xlabel("Month")
    plt.grid(True, alpha=0.3); plt.legend()
    pretty_date_axis(plt.gca())
    plt.tight_layout(); plt.show()

#===============================================================
# 8) “WHICH VERSION IS LOWEST?” — head-to-head frequency
#    For each month, which version yields the *lowest* EL_sen_30?
#===============================================================
""" metric = "EL_sen_30"
mat = []
labels = []
for label, suf in VERSIONS.items():
    c = col(metric, suf)
    if c:
        mat.append(df_all[c].values)
        labels.append(label)
if len(mat) >= 2:
    M = np.vstack(mat).T  # shape: (T, K)
    m_idx = np.nanargmin(M, axis=1)  # index of version with min EL each month
    counts = pd.Series(m_idx).value_counts().sort_index()
    share = (counts / counts.sum()).reindex(range(len(labels)), fill_value=0.0)
    share.index = labels

    plt.figure(figsize=(9.8,5.4))
    sns.barplot(x=share.index, y=share.values)
    plt.title("Share of Months with Lowest Senior EL @ 0.30")
    plt.ylabel("Share of months"); plt.xlabel("")
    plt.ylim(0, 1)
    plt.tight_layout(); plt.show()
 """
#===============================================================
# 9) DRAWDOWNS — Safe-asset multiplier (optimal s*)
#===============================================================
def compute_drawdown(x):
    x = pd.Series(x).copy()
    peak = x.cummax()
    dd = (x/peak) - 1.0
    return dd

#metric = "safe_asset_multiplier_optimal"
#plt.figure(figsize=(11,5.2))
#for label, suf in VERSIONS.items():
    c = col(metric, suf)
    if c:
        dd = compute_drawdown(df_all[c])
        plt.plot(dd.index, dd, label=label, alpha=0.9)
#plt.title("Drawdowns — Safe-Asset Multiplier (Optimal $s^*$)")
#plt.ylabel("Drawdown"); plt.xlabel("Month")
#plt.grid(True, alpha=0.3); plt.legend()
#pretty_date_axis(plt.gca())
#plt.tight_layout(); plt.show()

#===============================================================
# 10) SUMMARY BAR — mean, std, skew, kurt of EL_sen_30
#===============================================================
from scipy.stats import skew, kurtosis
""" 
stats_rows = []
for label, suf in VERSIONS.items():
    c = col("EL_sen_30", suf)
    if c:
        s = pd.Series(df_all[c]).dropna()
        if len(s):
            stats_rows.append({
                "Version": label,
                "Mean": s.mean(),
                "Std": s.std(),
                "Skew": skew(s),
                "Kurt": kurtosis(s, fisher=True),
            })
stats_df = pd.DataFrame(stats_rows)
if not stats_df.empty:
    fig, axes = plt.subplots(2,2, figsize=(12,8))
    for ax, m in zip(axes.flatten(), ["Mean","Std","Skew","Kurt"]):
        sns.barplot(data=stats_df, x="Version", y=m, ax=ax)
        ax.set_title(f"EL_sen_30 — {m} by Version"); ax.set_xlabel(""); ax.set_ylabel(m)
        for tick in ax.get_xticklabels(): tick.set_rotation(0)
    plt.tight_layout(); plt.show()
 """
#%%
#%% ---------- Multiple Bar Charts: One per variable ----------
metrics = [
    "optimal_sub",
    "EL_sen_30",
    "EL_jun_optimal",
    "safe_asset_multiplier_optimal",
    "safe_asset_multiplier_30",
    "PoolLoss_p99"
]

for metric in metrics:
    rows = []
    for label, suf in VERSIONS.items():
        colname = f"{metric}_{suf}"
        if colname in df_all.columns:
            mean_val = df_all[colname].mean()
            rows.append({"Version": label, "Mean": mean_val})
    
    if not rows:  # skip if no data for this metric
        continue
    
    summary_df = pd.DataFrame(rows)
    
    plt.figure(figsize=(8,5))
    sns.barplot(
        data=summary_df,
        x="Version", y="Mean",
        palette="Set2"
    )
    plt.title(f"Mean {metric} across Versions")
    plt.ylabel("Mean Value")
    plt.xlabel("")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.show()


#%% ================== Copula vs Copula(λ=0.50) — Comparison Suite ==================
from scipy.stats import ks_2samp, skew, kurtosis

sns.set_context("talk", rc={"lines.linewidth":1.6})
sns.set_style("whitegrid")

def _col(metric, suf):
    c = f"{metric}_{suf}"
    return c if c in df_all.columns else None

# Metrics to compare (adapt to your columns)
metrics = [
    "EL_sen_30",
    "EL_jun_optimal",
    "optimal_sub",
    "safe_asset_multiplier_optimal",
    "safe_asset_multiplier_30",
    "PoolLoss_p99",
]

suf_opt   = "Copula"      # optimized lambda version suffix
suf_fix   = "Copula050"   # fixed lambda=0.50 suffix
lab_opt   = "Copula (opt λ)"
lab_fix   = "Copula (λ=0.50)"

#-----------------------------
# 0) Small helper for ECDF
#-----------------------------
def ecdf_series(x):
    x = pd.Series(x).dropna().sort_values().values
    n = len(x)
    if n == 0: 
        return np.array([]), np.array([])
    y = np.arange(1, n+1) / n
    return x, y

#-----------------------------
# 1) TIME-SERIES OVERLAYS
#-----------------------------
for metric in metrics:
    c_opt = _col(metric, suf_opt)
    c_fix = _col(metric, suf_fix)
    if not (c_opt and c_fix): 
        continue

    plt.figure(figsize=(11,5.2))
    plt.plot(df_all.index, df_all[c_opt], label=lab_opt)
    plt.plot(df_all.index, df_all[c_fix], label=lab_fix)
    plt.title(f"{metric} — Time Series Overlay")
    plt.ylabel(metric); plt.xlabel("Month"); plt.grid(True, alpha=0.3); plt.legend()
    try: pretty_date_axis(plt.gca())
    except: pass
    plt.tight_layout(); plt.show()

#-----------------------------
# 2) ECDF COMPARISON
#-----------------------------
for metric in metrics:
    c_opt = _col(metric, suf_opt)
    c_fix = _col(metric, suf_fix)
    if not (c_opt and c_fix): 
        continue

    x_opt, y_opt = ecdf_series(df_all[c_opt])
    x_fix, y_fix = ecdf_series(df_all[c_fix])

    if len(x_opt) and len(x_fix):
        plt.figure(figsize=(8.5,5.2))
        plt.step(x_opt, y_opt, where="post", label=lab_opt)
        plt.step(x_fix, y_fix, where="post", label=lab_fix)
        plt.title(f"ECDF — {metric}")
        plt.xlabel(metric); plt.ylabel("F(x)")
        plt.grid(True, alpha=0.3); plt.legend()
        plt.tight_layout(); plt.show()

#-----------------------------
# 3) QQ PLOTS (opt vs fixed)
#-----------------------------
""" def qqplot(a, b, label_a, label_b, metric):
    s0 = pd.Series(a).dropna().sort_values().values
    s1 = pd.Series(b).dropna().sort_values().values
    n = min(len(s0), len(s1))
    if n < 10: 
        return
    s0 = s0[:n]; s1 = s1[:n]
    plt.figure(figsize=(6.2,6))
    plt.scatter(s0, s1, s=12, alpha=0.6)
    lo = float(np.nanmin([s0.min(), s1.min()])); hi = float(np.nanmax([s0.max(), s1.max()]))
    plt.plot([lo, hi], [lo, hi], ls="--", lw=1.2)
    plt.title(f"QQ Plot — {label_a} vs {label_b}\n({metric})")
    plt.xlabel(f"{label_a} quantiles"); plt.ylabel(f"{label_b} quantiles")
    plt.tight_layout(); plt.show()

#for metric in metrics:
    c_opt = _col(metric, suf_opt)
    c_fix = _col(metric, suf_fix)
    if not (c_opt and c_fix): 
        continue
    qqplot(df_all[c_opt], df_all[c_fix], lab_opt, lab_fix, metric)
 """
#-----------------------------
# 4) SCATTER WITH 45° LINE
#-----------------------------
""" for metric in metrics:
    c_opt = _col(metric, suf_opt)
    c_fix = _col(metric, suf_fix)
    if not (c_opt and c_fix): 
        continue
    x = df_all[c_opt]; y = df_all[c_fix]
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 10: 
        continue
    plt.figure(figsize=(6.4,6))
    plt.scatter(x[m], y[m], s=14, alpha=0.6)
    lo = float(np.nanmin([x[m].min(), y[m].min()])); hi = float(np.nanmax([x[m].max(), y[m].max()]))
    plt.plot([lo, hi], [lo, hi], ls="--", lw=1.2, label="45° line")
    plt.title(f"{metric} — {lab_opt} vs {lab_fix}")
    plt.xlabel(lab_opt); plt.ylabel(lab_fix)
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout(); plt.show()
 """
#-----------------------------
# 5) ROLLING SPREAD & RATIO
#-----------------------------
win = 12
for metric in metrics:
    c_opt = _col(metric, suf_opt)
    c_fix = _col(metric, suf_fix)
    if not (c_opt and c_fix): 
        continue
    spread = df_all[c_opt] - df_all[c_fix]
    ratio  = df_all[c_opt] / df_all[c_fix]
    plt.figure(figsize=(11,5.2))
    plt.plot(spread.index, spread.rolling(win).mean(), label=f"Rolling {win}M mean spread (opt - fixed)")
    plt.axhline(0, color="k", lw=1, ls="--")
    plt.title(f"{metric} — Rolling Spread ({lab_opt} - {lab_fix})")
    plt.ylabel("Spread"); plt.xlabel("Month"); plt.grid(True, alpha=0.3); plt.legend()
    try: pretty_date_axis(plt.gca())
    except: pass
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(11,5.2))
    plt.plot(ratio.index, ratio.rolling(win).median(), label=f"Rolling {win}M median ratio (opt / fixed)")
    plt.axhline(1.0, color="k", lw=1, ls="--")
    plt.title(f"{metric} — Rolling Ratio ({lab_opt} / {lab_fix})")
    plt.ylabel("Ratio"); plt.xlabel("Month"); plt.grid(True, alpha=0.3); plt.legend()
    try: pretty_date_axis(plt.gca())
    except: pass
    plt.tight_layout(); plt.show()

#-----------------------------
# 6) HISTOGRAM OF DIFFERENCES
#-----------------------------
""" for metric in metrics:
    c_opt = _col(metric, suf_opt)
    c_fix = _col(metric, suf_fix)
    if not (c_opt and c_fix): 
        continue
    diff = (df_all[c_opt] - df_all[c_fix]).dropna()
    if not len(diff): 
        continue
    plt.figure(figsize=(9,5))
    plt.hist(diff, bins=40, alpha=0.7)
    plt.title(f"Histogram of Differences — {metric}\n({lab_opt} - {lab_fix})")
    plt.xlabel("Difference"); plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.show()
 """
#-----------------------------
# 7) SUMMARY STATS BAR (mean/std/skew/kurt) + KS test
#-----------------------------
""" for metric in metrics:
    c_opt = _col(metric, suf_opt)
    c_fix = _col(metric, suf_fix)
    if not (c_opt and c_fix): 
        continue
    a = pd.Series(df_all[c_opt]).dropna()
    b = pd.Series(df_all[c_fix]).dropna()
    if len(a) < 5 or len(b) < 5:
        continue

    stats_df = pd.DataFrame([
        {"Version": lab_opt, "Mean": a.mean(), "Std": a.std(), "Skew": skew(a), "Kurt": kurtosis(a, fisher=True)},
        {"Version": lab_fix, "Mean": b.mean(), "Std": b.std(), "Skew": skew(b), "Kurt": kurtosis(b, fisher=True)},
    ])

    fig, axes = plt.subplots(2,2, figsize=(11,7))
    for ax, m in zip(axes.flatten(), ["Mean","Std","Skew","Kurt"]):
        sns.barplot(data=stats_df, x="Version", y=m, hue="Version", palette="Set2", legend=False, ax=ax)
        ax.set_title(f"{metric} — {m}"); ax.set_xlabel(""); ax.set_ylabel(m)
    plt.tight_layout(); plt.show()

    # KS test
    stat, pval = ks_2samp(a.values, b.values)
    print(f"[KS] {metric}: stat={stat:.4f}, p-value={pval:.4f}  ({lab_opt} vs {lab_fix})")

 """
#%%

#%% PLOTS 3, 4, and 9 using saved EL_sen_XX 
sns.set_context("talk")
sns.set_style("whitegrid")

# Subordination checkpoints you saved
_s_levels = [0, 10, 20, 30, 40, 50, 60, 70]              # as percentages
_s_cols   = [f"{pp:02d}" for pp in _s_levels]            # "00","10",...

# -------------------------- Plot 3: EL(s) curve (time-avg) with IQR --------------------------
plt.figure(figsize=(11,6))
plotted_any = False
for label, suf in VERSIONS.items():
    cols = [f"EL_sen_{pp:02d}_{suf}" for pp in _s_levels]
    have = [c for c in cols if c in df_all.columns]
    if len(have) < 3:
        continue

    # Time-aggregate: mean and IQR across months
    M  = df_all[have].mean(axis=0).to_numpy()
    Q1 = df_all[have].quantile(0.25, axis=0).to_numpy()
    Q3 = df_all[have].quantile(0.75, axis=0).to_numpy()

    # x-axis from the column names we have (keeps alignment even if some are missing)
    xs = [int(c.split("_")[2]) / 100.0 for c in have]
    order = np.argsort(xs)
    xs = np.array(xs)[order]; M = M[order]; Q1 = Q1[order]; Q3 = Q3[order]

    plt.plot(xs, M, marker="o", linewidth=2, label=label)
    plt.fill_between(xs, Q1, Q3, alpha=0.15, linewidth=0)
    plotted_any = True

# Optional horizontal reference: average Bund EL (from MC columns if present)
if "EL_germany_MC" in df_all.columns and plotted_any:
    plt.axhline(df_all["EL_germany_MC"].mean(), ls="--", alpha=0.6, label="German Bund (avg)")

if plotted_any:
    plt.title("Senior Expected Loss vs Subordination (mean across months, IQR shaded)")
    plt.xlabel("Subordination s")
    plt.ylabel("Senior EL")
    plt.xticks([p/100 for p in _s_levels], [f"{p}%" for p in _s_levels])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("[Plot 3] No EL_sen_XX columns found. Skipped.")

# ------------------ Plot 4: Time × Subordination heatmaps (one per version) ------------------
""" for label, suf in VERSIONS.items():
    cols = [f"EL_sen_{pp:02d}_{suf}" for pp in _s_levels if f"EL_sen_{pp:02d}_{suf}" in df_all.columns]
    if len(cols) < 3:
        continue

    mat = df_all[cols].copy()

    # ensure a proper Month-End DatetimeIndex (pandas alias is "M", not "ME")
    try:
        mat = mat.asfreq("M")
    except Exception:
        pass
    mat.index = pd.to_datetime(mat.index)

    # columns -> numeric s in % for axis labels (0, 10, …, 70)
    mat.columns = [int(c.split("_")[2]) for c in mat.columns]

    # --- plotting ---
    plt.figure(figsize=(11.5, 4.8))
    ax = sns.heatmap(
        mat.T,
        cmap="viridis",
        cbar_kws={"label": "Senior EL"},
        xticklabels=False  # we'll set clean labels right below
    )

    # build nice YYYY-MM labels with a sensible step
    month_labels = mat.index.strftime("%Y-%m")
    # choose about ~12 ticks across the axis
    step = max(len(month_labels) // 12, 1)
    tick_pos = np.arange(0.5, len(month_labels) + 0.5, step)

    ax.set_xticks(tick_pos)
    ax.set_xticklabels(month_labels[::step], rotation=45, ha="right")

    ax.set_title(f"Heatmap — Senior EL across time and s ({label})")
    ax.set_xlabel("Year–Month")
    ax.set_ylabel("Subordination s (%)")

    # y ticks are the subordination levels; render as percentages
    ax.set_yticklabels([f"{int(s)}%" for s in mat.columns])

    plt.tight_layout()
    plt.show()

# ---------------- Plot 9: ECDF of Junior EL at optimal s* (per version) ----------------
#long_rows = []
#for label, suf in VERSIONS.items():
    colname = f"EL_jun_optimal_{suf}"
    if colname in df_all.columns:
        s = df_all[colname].dropna()
        if len(s):
            long_rows.append(pd.DataFrame({"EL_jun_optimal": s.values, "Version": label}))

#if long_rows:
    ecdf_df = pd.concat(long_rows, ignore_index=True)
    plt.figure(figsize=(11,5.6))
    sns.ecdfplot(data=ecdf_df, x="EL_jun_optimal", hue="Version")
    plt.title("ECDF — Junior Expected Loss at Optimal $s^*$")
    plt.xlabel("EL (junior, at $s^*$)")
    plt.ylabel("F(x)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
#else:
    print("[Plot 9] No EL_jun_optimal_* columns found. Skipped.")
 """
#%%
# --- styling ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("talk")
sns.set_style("whitegrid")

# Assumptions:
# - df_all is your DataFrame with monthly rows.
# - VERSIONS is a dict like {"MC baseline":"MC", ...}
# - Senior EL columns are named "EL_sen_{ss}_{suf}" where ss in {"00","10",...}
# - German Bund EL column for MC is "EL_germany_MC"

# Subordination checkpoints you saved
_s_levels = [0, 10, 20, 30, 40, 50, 60, 70]              # as percentages
_s_cols   = [f"{pp:02d}" for pp in _s_levels]            # "00","10",...

# ---- choose only the MC baseline version ----
# Try a few robust matches to find the MC baseline suffix
mc_keys = []
for label, suf in VERSIONS.items():
    ll = label.lower()
    if ("mc" in ll and "base" in ll) or suf.upper() == "MC":
        mc_keys.append((label, suf))

if not mc_keys:
    raise ValueError("Couldn't find an MC baseline in VERSIONS. "
                     "Make sure VERSIONS contains something like {'MC baseline':'MC'}.")

label, suf = mc_keys[0]  # take the first match

# --------------------- Plot: EL(s) curve (time-avg) for MC baseline ---------------------
#%%
plt.figure(figsize=(11, 6))

cols = [f"EL_sen_{pp:02d}_{suf}" for pp in _s_levels]
have = [c for c in cols if c in df_all.columns]
if len(have) < 3:
    raise ValueError(f"Not enough EL_sen_* columns for suffix '{suf}'. Found: {have}")

# Time-aggregate: mean and IQR across months
M  = df_all[have].mean(axis=0).to_numpy()
#Q1 = df_all[have].quantile(0.25, axis=0).to_numpy()
#Q3 = df_all[have].quantile(0.75, axis=0).to_numpy()

# x-axis from the column names we have (keeps alignment even if some are missing)
xs = [int(c.split("_")[2]) / 100.0 for c in have]
order = np.argsort(xs)
xs = np.array(xs)[order]; M = M[order]; Q1 = Q1[order]; Q3 = Q3[order]

plt.plot(xs, M, marker="o", linewidth=2, label=label)
#plt.fill_between(xs, Q1, Q3, alpha=0.15, linewidth=0)

# ---- horizontal references ----
# (1) Avg German Bund EL (if present)
if "EL_germany_MC" in df_all.columns:
    bund_avg = df_all["EL_germany_MC"].mean()
    plt.axhline(bund_avg, ls="--", alpha=0.7, label="German Bund (avg)")

# (2) AAA threshold at 0.5% EL
AAA_EL = 0.005  # 0.5%
plt.axhline(AAA_EL, ls=":", alpha=0.9, linewidth=2, label="AAA threshold (0.5% EL)")

# ---- cosmetics ----
plt.title("Senior Expected Loss vs Subordination — MC baseline (mean across months, IQR shaded)")
plt.xlabel("Subordination s")
plt.ylabel("Senior EL")
plt.xticks([p/100 for p in _s_levels], [f"{p}%" for p in _s_levels])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

#%%
#%% Senior EL vs Subordination — t-Copula (opt λ), mean across months + IQR
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("talk")
sns.set_style("whitegrid")

# subordination checkpoints you saved
_s_levels = [0, 10, 20, 30, 40, 50, 60, 70]
_s_cols   = [f"{pp:02d}" for pp in _s_levels]

# robustly pick the Copula (optimized λ) suffix & label
copula_keys = []
for label, suf in VERSIONS.items():
    if suf == "Copula" or ("copula" in label.lower() and "opt" in label.lower()):
        copula_keys.append((label, suf))
if not copula_keys:
    raise ValueError("Couldn't find the Copula (opt λ) version in VERSIONS.")
label_cop, suf_cop = copula_keys[0]

plt.figure(figsize=(11, 6))

# collect available EL_sen_{pp}_{Copula} columns
cols = [f"EL_sen_{pp:02d}_{suf_cop}" for pp in _s_levels]
have = [c for c in cols if c in df_all.columns]
if len(have) < 3:
    raise ValueError(f"Not enough EL_sen_* columns for suffix '{suf_cop}'. Found: {have}")

# time aggregates
M  = df_all[have].mean(axis=0).to_numpy()
#Q1 = df_all[have].quantile(0.25, axis=0).to_numpy()
#Q3 = df_all[have].quantile(0.75, axis=0).to_numpy()

# x from the column names we actually have (keeps order correct)
xs = [int(c.split("_")[2]) / 100.0 for c in have]
order = np.argsort(xs)
xs = np.array(xs)[order]; M = M[order]; Q1 = Q1[order]; Q3 = Q3[order]

# plot mean + IQR
plt.plot(xs, M, marker="o", linewidth=2.2, label=f"{label_cop}")
#plt.fill_between(xs, Q1, Q3, alpha=0.20, linewidth=0)

# horizontal references
if "EL_germany_MC" in df_all.columns:
    bund_avg = df_all["EL_germany_MC"].mean()
    plt.axhline(bund_avg, ls="--", alpha=0.7, label="German Bund (avg)")
AAA_EL = 0.005  # 0.5%
plt.axhline(AAA_EL, ls=":", alpha=0.9, linewidth=2, label="AAA threshold (0.5% EL)")

# cosmetics
plt.title("Senior Expected Loss vs Subordination — t-Copula (opt λ) (mean across months))")
plt.xlabel("Subordination s")
plt.ylabel("Senior EL")
plt.xticks([p/100 for p in _s_levels], [f"{p}%" for p in _s_levels])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %%
#%% ---------- Summary chart: ELs (senior/junior at s* and at 30%) + PoolLoss_p99 ----------
import matplotlib.ticker as mtick

# (model label, suffix in df_all)
models = [
    ("t-Copula (opt λ)", "Copula"),
    ("Copula + Contagion", "AdvCon"),
    ("Crisis Regime (Stress)", "Stress"),
]

rows = []
p99_points = []

for label, suf in models:
    cols_required = {
        "EL_sen_opt": f"EL_sen_optimal_{suf}",
        "EL_jun_opt": f"EL_jun_optimal_{suf}",
        "EL_sen_30":  f"EL_sen_30_{suf}",
        "EL_jun_30":  f"EL_jun_30_{suf}",
        "Pool_p99":   f"PoolLoss_p99_{suf}",
    }
    # Skip model if any core column is missing
    if not all(c in df_all.columns for c in cols_required.values()):
        print(f"[warn] Missing columns for {label}; skipping. Needed:", cols_required)
        continue

    # means across months (drop NaNs)
    sen_opt = df_all[cols_required["EL_sen_opt"]].dropna().mean()
    jun_opt = df_all[cols_required["EL_jun_opt"]].dropna().mean()
    sen_30  = df_all[cols_required["EL_sen_30"]].dropna().mean()
    jun_30  = df_all[cols_required["EL_jun_30"]].dropna().mean()
    p99     = df_all[cols_required["Pool_p99"]].dropna().mean()

    rows += [
        {"Model": label, "Metric": "Senior @ optimal s*", "Value": sen_opt},
        {"Model": label, "Metric": "Junior @ optimal s*", "Value": jun_opt},
        {"Model": label, "Metric": "Senior @ s=30%",      "Value": sen_30},
        {"Model": label, "Metric": "Junior @ s=30%",      "Value": jun_30},
    ]
    p99_points.append({"Model": label, "PoolLoss_p99": p99})

summary = pd.DataFrame(rows)
p99_df  = pd.DataFrame(p99_points)

# Order metrics in legend
metric_order = ["Senior @ optimal s*", "Junior @ optimal s*",
                "Senior @ s=30%", "Junior @ s=30%"]

plt.figure(figsize=(12,6))
ax1 = plt.gca()
sns.barplot(data=summary, x="Model", y="Value", hue="Metric",
            order=[m[0] for m in models if m[0] in summary["Model"].unique()],
            hue_order=metric_order, ax=ax1)

# Format as percentages (your ELs are in fractions)
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax1.set_ylabel("Expected loss (%)")
ax1.set_xlabel("")
ax1.set_title("Senior & Junior ELs — optimal s* vs s=30% (means across months)\n"
              "+ PoolLoss p99 (secondary axis)")

# Secondary axis for PoolLoss p99 (also % scale for clarity)
ax2 = ax1.twinx()
if not p99_df.empty:
    # Align x positions with bar clusters
    x_pos = np.arange(len(p99_df))
    ax2.plot(x_pos, p99_df["PoolLoss_p99"].values, marker="o", linewidth=2, label="PoolLoss p99")
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.set_ylabel("PoolLoss p99 (%)")
    # sync x tick labels with models
    ax1.set_xticklabels(p99_df["Model"].tolist(), rotation=10)

# Legends: combine from both axes
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", ncol=2, frameon=True)

plt.tight_layout()
plt.show()


#%%---------------------------------------------------------------


#%% ---------- Senior & Junior ELs at s = {10,30,50,70} for each model ----------
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# which subordination checkpoints to show
subs_pp = [10, 30, 50, 70]  # percentages
subs_keys = [f"{pp:02d}" for pp in subs_pp]

def _safe_col(df, name):
    return name if name in df.columns else None

# Build a tidy frame with Senior & Junior ELs (both normalized to tranche notional)
rows = []
for label, suf in VERSIONS.items():
    # Skip MC and lambda=0.50 versions
    if label == "MC" or "λ=0.50" in label:
        continue
    
    pool_col = _safe_col(df_all, f"EL_pool_{suf}")
    if not pool_col:
        print(f"[warn] Missing EL_pool for {label}; skipping this version.")
        continue
    
    for pp, key in zip(subs_pp, subs_keys):
        sen_col = _safe_col(df_all, f"EL_sen_{key}_{suf}")
        if not sen_col:
            continue
        
        s = pp / 100.0  # subordination as decimal
        
        # Get the time series
        sen_norm = df_all[sen_col]      # senior EL normalized to (1-s)
        pool = df_all[pool_col]         # pool EL per pool notional
        
        # Convert senior back to pool-based loss
        sen_pool = sen_norm * (1 - s)   # senior loss per pool notional
        
        # Junior pool-based loss
        jun_pool = pool - sen_pool      # junior loss per pool notional
        
        # Normalize junior to its tranche notional
        jun_norm = jun_pool / s         # junior EL normalized to s
        
        # Sanity check: conservation property
        check_pool = sen_norm * (1-s) + jun_norm * s
        if not np.allclose(check_pool.dropna(), pool.dropna(), atol=1e-6):
            print(f"⚠️  Conservation violated for {label}, s={pp}%")
            print(f"    Pool EL mean: {pool.dropna().mean():.6f}")
            print(f"    Reconstructed: {check_pool.dropna().mean():.6f}")
        
        # Store means (or use .median() if you prefer robustness)
        rows.append({
            "Model": label, 
            "Subordination": f"s={pp}%",
            "Tranche": "Senior", 
            "EL": sen_norm.dropna().mean()
        })
        rows.append({
            "Model": label, 
            "Subordination": f"s={pp}%",
            "Tranche": "Junior", 
            "EL": jun_norm.dropna().mean()
        })

el_grid = pd.DataFrame(rows)

if el_grid.empty:
    print("[EL grid] No data found for requested subordination checkpoints.")
else:
    # Ensure consistent order across facets (excluding MC and λ=0.50)
    model_order = [m for m, s in VERSIONS.items() 
                   if m in el_grid["Model"].unique() and m != "MC" and "λ=0.50" not in m]
    tranche_order = ["Senior", "Junior"]
    sub_order = [f"s={pp}%" for pp in subs_pp if f"s={pp}%" in el_grid["Subordination"].unique()]
    
    # Plot: one subplot per subordination level
    n_sub = len(sub_order)
    ncols = min(2, n_sub)
    nrows = int(np.ceil(n_sub / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.8*nrows), squeeze=False)
    axes = axes.flatten()
    
    for i, sub in enumerate(sub_order):
        ax = axes[i]
        data_i = el_grid[el_grid["Subordination"] == sub]
        
        sns.barplot(
            data=data_i, x="Model", y="EL", hue="Tranche",
            order=model_order, hue_order=tranche_order,
            ax=ax
        )
        
        ax.set_title(f"Expected Loss by Tranche — {sub}")
        ax.set_xlabel("")
        ax.set_ylabel("Expected loss (%)")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        for tick in ax.get_xticklabels(): 
            tick.set_rotation(12)
        ax.legend(loc="upper right", frameon=True)
    
    # Hide any unused axes
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary to verify Junior >= Senior
    print("\n" + "="*60)
    print("SUMMARY: Junior vs Senior EL (both normalized to tranche)")
    print("="*60)
    for sub in sub_order:
        print(f"\n{sub}:")
        data_sub = el_grid[el_grid["Subordination"] == sub]
        for model in model_order:
            data_model = data_sub[data_sub["Model"] == model]
            if len(data_model) == 2:
                sen_val = data_model[data_model["Tranche"] == "Senior"]["EL"].values[0]
                jun_val = data_model[data_model["Tranche"] == "Junior"]["EL"].values[0]
                status = "✓" if jun_val >= sen_val else "⚠️ VIOLATION"
                print(f"  {model:20s}: Senior={sen_val:7.4%}, Junior={jun_val:7.4%}  {status}")
# %%
#%% ---------- Pool Expected Loss over time (Copula / AdvCon / Stress) with custom shaded crises ----------
import matplotlib.ticker as mtick
from matplotlib.patches import Patch

def shade_custom_crises(ax):
    # (label, start, end, color)
    spans = [
        ("Euro Area Debt Crisis", pd.Timestamp("2010-01-01"), pd.Timestamp("2013-12-31"), "#ffd6d6"),  # light red
        ("COVID-19 Shock",        pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31"), "#ffe5cc"),  # light orange
        ("Energy/Inflation",      pd.Timestamp("2021-01-01"), pd.Timestamp("2023-12-31"), "#fff9cc"),  # light yellow
    ]
    handles = []
    for lab, start, end, col in spans:
        ax.axvspan(start, end, color=col, alpha=1.0, zorder=0)
        handles.append(Patch(facecolor=col, edgecolor="none", label=lab))
    return handles

models = [
    ("Copula", "t-Copula (opt λ)"),
    ("AdvCon", "Copula + Contagion"),
    ("Stress", "Crisis Regime (Stress)"),
]

plt.figure(figsize=(12, 5.2))
plotted = False
for suf, lab in models:
    col = f"EL_pool_{suf}"
    if col in df_all.columns:
        plt.plot(df_all.index, df_all[col], label=lab, linewidth=1.9)
        plotted = True

# Shade crises
shade_handles = shade_custom_crises(plt.gca())

plt.title("Pool Expected Loss — Crisis Windows Shaded")
plt.ylabel("Pool Expected Loss")
plt.xlabel("Month")
plt.grid(True, alpha=0.3)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# Combine model legend with shaded-window legend
if plotted:
    model_handles, model_labels = plt.gca().get_legend_handles_labels()
    plt.legend(model_handles + shade_handles, model_labels + [h.get_label() for h in shade_handles],
               loc="upper left", ncol=2, frameon=True)

try:
    pretty_date_axis(plt.gca())
except Exception:
    pass

plt.tight_layout()
plt.show()


#%%
#%% ---------- Safe-Asset Multiplier — time series for selected windows (3 models only) ----------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tseries.offsets import MonthEnd

sns.set_context("talk")
sns.set_style("whitegrid")

# Models to keep
MODELS = [
    ("Copula", "t-Copula (opt λ)"),
    ("AdvCon", "Copula + Contagion"),
    ("Stress", "Crisis Regime (Stress)"),
]

METRICS = [
    ("safe_asset_multiplier_30",      "s = 30%"),
    ("safe_asset_multiplier_optimal", "optimal s*"),
]

def _window_bounds(start_str, end_str):
    """Return inclusive [start, end] snapped to month ends (end inclusive)."""
    s = pd.Timestamp(start_str)
    e = pd.Timestamp(end_str)
    # snap start to month start’s month-end if your index is month-end; otherwise keep s as-is
    s_me = (s + MonthEnd(0)) if s.day != 1 else (s + MonthEnd(0))  # safe for month-end indices
    e_me = e + MonthEnd(0)  # ensures inclusive upper bound at month-end
    return s_me.normalize(), e_me.normalize()

def plot_and_print(df_all, start, end, title_suffix):
    # Build long-form for the three models & two metrics
    rows = []
    for suf, label in MODELS:
        for metric, mlabel in METRICS:
            col = f"{metric}_{suf}"
            if col not in df_all.columns:
                print(f"[warn] Missing column: {col}")
                continue
            s = df_all[[col]].copy()
            s.columns = ["Value"]
            s["Month"]  = s.index
            s["Model"]  = label
            s["Metric"] = mlabel
            rows.append(s)
    if not rows:
        print("[error] No matching columns found.")
        return

    L = pd.concat(rows, ignore_index=True)

    # Robust, inclusive window slice + explicit x-limits
    lo, hi = _window_bounds(start, end)
    mask = (L["Month"] >= lo) & (L["Month"] <= hi)
    L = L.loc[mask].copy()

    # --- Time series only ---
    plt.figure(figsize=(12, 5.6))
    for (label, mlabel), g in L.groupby(["Model", "Metric"]):
        g = g.sort_values("Month")
        plt.plot(g["Month"], g["Value"], linewidth=1.9, label=f"{label} — {mlabel}")
    plt.title(f"Safe-Asset Multiplier — {title_suffix}")
    plt.ylabel("Multiplier (unitless)")
    plt.xlabel("Month")
    plt.grid(True, alpha=0.3)
    ax = plt.gca()
    ax.set_xlim(lo, hi)  # <- prevents any spillover past 2014-12-31, etc.
    try:
        pretty_date_axis(ax)
    except Exception:
        pass
    plt.legend(ncol=2, loc="upper left", frameon=True)
    plt.tight_layout()
    plt.show()

    # --- Printed window averages (no bars) ---
    avg = (L.groupby(["Model","Metric"])["Value"]
             .mean()
             .reset_index()
             .sort_values(["Metric","Model"]))
    print("\n== Window Averages:", title_suffix, "==")
    for metric_name in avg["Metric"].unique():
        print(f"\n  {metric_name}:")
        sub = avg[avg["Metric"] == metric_name]
        for _, row in sub.iterrows():
            print(f"    {row['Model']:<22s}  {row['Value']:.3f}")

# ----- Figure A: Jan 2010 to Dec 2014 (hard stop at 2014-12-31) -----
plot_and_print(df_all, "2010-01-01", "2014-12-31", "Jan 2010 – Dec 2014")

# ----- Figure B: Mar 2020 to Dec 2021 -----
plot_and_print(df_all, "2020-02-01", "2021-12-31", "Mar 2020 – Dec 2021")

#Figure C : Jan 2021 to Dec 2023
plot_and_print(df_all, "2021-01-01", "2023-12-31", "Jan 2021 – Dec 2023")
# %%

#------------------------------------------------------------------------------------------------------------------------------------

#FORMATTED USEFUL GRAPHS

#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.ticker import PercentFormatter, ScalarFormatter

# ===============================
# Global minimalist B&W journal style
# ===============================
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
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "grid.color": "0.85",
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
    "axes.prop_cycle": cycler("color", ["black"]),
    "axes.formatter.useoffset": False,
    "axes.formatter.use_mathtext": False,
})

# -------------------------------
# Inputs expected in the session:
# - df_all: DataFrame with EL_sen_{pp}_{suffix} columns and (optionally) 'EL_germany_MC'
# - VERSIONS: dict mapping labels -> suffix strings
# -------------------------------

_s_levels = [0, 10, 20, 30, 40, 50, 60, 70]

# Pick Copula (opt λ)
copula_keys = []
for label, suf in VERSIONS.items():
    if (suf == "Copula") or ("copula" in str(label).lower() and "opt" in str(label).lower()):
        copula_keys.append((label, suf))
if not copula_keys:
    raise ValueError("Couldn't find the Copula (opt λ) version in VERSIONS.")
label_cop, suf_cop = copula_keys[0]

cols = [f"EL_sen_{pp:02d}_{suf_cop}" for pp in _s_levels]
have = [c for c in cols if c in df_all.columns]
if len(have) < 3:
    raise ValueError(f"Not enough EL_sen_* columns for suffix '{suf_cop}'. Found: {have}")

M = df_all[have].mean(axis=0).to_numpy()
xs = [int(c.split("_")[2]) / 100.0 for c in have]
order = np.argsort(xs)
xs = np.array(xs)[order]
M  = M[order]
is_decimal = np.nanmax(M) <= 1.0

# -------------------------------
# Plot
# -------------------------------
fig, ax = plt.subplots(figsize=(10.8, 5.8))

ax.plot(xs, M, linestyle="-", linewidth=1.8, marker="o", markersize=3.5,
        label=f"{label_cop}")

if "EL_germany_MC" in df_all.columns:
    bund_avg = float(df_all["EL_germany_MC"].mean())
    ax.axhline(bund_avg, linestyle="--", linewidth=1.0, color="black", label="German Bund (avg)")

AAA_EL = 0.005  # 0.5%
ax.axhline(AAA_EL, linestyle=":", linewidth=1.2, color="black", label="AAA threshold (0.5% EL)")

ax.set_xlabel("Subordination level (%)", fontsize=12)
ax.set_ylabel("Senior EL (%)" if is_decimal else "Senior EL", fontsize=12)
ax.set_xticks([p/100 for p in _s_levels])
ax.set_xticklabels([f"{p}%" for p in _s_levels], fontsize=12)

# Percent labels with no decimals
if is_decimal:
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
else:
    sf = ScalarFormatter(useMathText=False)
    sf.set_scientific(False)
    sf.set_useOffset(False)
    ax.yaxis.set_major_formatter(sf)
ax.get_yaxis().get_offset_text().set_visible(False)

# --- Ensure tick marks are visible exactly at current labels ---
# Force ticks ON for left/bottom; draw them in+out so you see them even if outer part is cropped
ax.tick_params(axis="x", which="major", bottom=True, top=False,
               length=6, width=0.9, direction="out")
ax.tick_params(axis="y", which="major", left=True, right=False,
               length=6, width=0.9, direction="out")

# Grid: horizontal only
ax.yaxis.grid(True)
ax.xaxis.grid(False)

# Minimal frame
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

leg = ax.legend(loc="best", frameon=True, fontsize=10,
                facecolor="white", edgecolor="black")
leg.get_frame().set_linewidth(0.8)

fig.tight_layout()

os.makedirs("FinalFigures", exist_ok=True)
out_path = "FinalFigures/SeniorEL_vs_Subordination_CopulaOpt.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight")

print(f"Saved: {out_path}")

# %%

#----------------------------------------
#FIG 16


#NEW GRAPHIC STYLE------------------------
# Academic-style Safe-Asset Multiplier plot (Fixed s = 0.30)
# - Single clean tick marks (no minors)
# - Monochrome black & white, Computer Modern font
# - Dotted horizontal gridlines only
# - Legend with thin border inside the plot

#Figure 16
import matplotlib.pyplot as plt
from cycler import cycler

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

    # Grid (only horizontal)
    "grid.color": "0.85",
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,

    # Monochrome line colors
    "axes.prop_cycle": cycler("color", ["black"]),
})

fig, ax = plt.subplots(figsize=(11, 5))

# Plot the Safe-Asset Multiplier series
ax.plot(
    df_all.index,
    df_all.get("safe_asset_multiplier_30_Copula"),
    linewidth=1.6,
    linestyle="-",
    label="Safe-asset multiplier (s = 0.30)"
)

# Labels and professional-style title
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Safe Asset Multiplier", fontsize=12)
ax.set_title("Safe-asset multiplier — fixed subordination s = 0.30", fontsize=12)

# Grid and frame
ax.yaxis.grid(True)
ax.xaxis.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Ticks: clean and balanced ---
ax.tick_params(axis="x",
               which="major",
               direction="out",
               length=4.0,
               width=0.7,
               labelsize=12,
               bottom=True, top=False)
ax.tick_params(axis="y",
               which="major",
               direction="out",
               length=4.0,
               width=0.7,
               labelsize=12,
               left=True, right=False)

# No minor ticks (clean look)
ax.tick_params(axis="both", which="minor", bottom=False, left=False)

# Legend inside with thin border
leg = ax.legend(
    loc="upper right",
    frameon=True,
    fontsize=10,
    facecolor="white",
    edgecolor="black"
)
leg.get_frame().set_linewidth(0.8)

# Apply your existing date formatter
pretty_date_axis(ax)

fig.tight_layout()
plt.savefig("FinalFigures/SafeAssetMultiplier_s30_Copula.png", dpi=300, bbox_inches="tight")
plt.show()

# %% Fig 17

# Academic-style Senior Expected Loss vs Subordination (mean across months)
# Updated legend names:
#   - Model 1 (baseline t-copula)
#   - Model 2 (Adverse contagion)
#   - Model 3 (Stress regime)
#
# Style: Monochrome B&W, serif font, dotted horizontal grid,
#        thin tick marks, clean axes.

import re
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.ticker import FuncFormatter

# ----------------------------- Style -----------------------------
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
    "grid.color": "0.85",
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
    "axes.prop_cycle": cycler("color", ["black"]),
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

# ----------------------------- Helpers -----------------------------
def _available_suffixes(df_all):
    suf = set()
    pat = re.compile(r"^EL_sen_\d{2}_(.+)$")
    for c in df_all.columns:
        m = pat.match(c)
        if m:
            suf.add(m.group(1))
    return suf

def _pick_suffix(df_all, candidates):
    have = _available_suffixes(df_all)
    for cand in candidates:
        if cand in have:
            return cand
    have_l = {h.lower(): h for h in have}
    for cand in candidates:
        for h_l, h in have_l.items():
            if cand.lower() in h_l:
                return h
    return None

# ----------------------------- Target series -----------------------------
target_series = {
    "Model 1 (baseline t-copula)": ["Copula"],
    "Model 2 (Adverse contagion)": ["AdvCon", "Adverse", "Contagion", "Adv"],
    "Model 3 (Stress regime)": ["Stress", "Crisis", "StressTest", "RegimeStress"],
}

# Resolve suffixes present in df_all
resolved = {}
for label, candidates in target_series.items():
    suf = _pick_suffix(df_all, candidates)
    if suf is not None:
        resolved[label] = suf
    else:
        print(f"[warn] No suffix found for '{label}' among candidates {candidates}. Skipping this line.")

# ----------------------------- Plot -----------------------------
_s_levels = [0, 10, 20, 30, 40, 50, 60, 70]
line_styles = {
    "Model 1 (baseline t-copula)": ("-", 1.8),
    "Model 2 (Adverse contagion)": ("--", 1.6),
    "Model 3 (Stress regime)": (":", 1.6),
}

plt.figure(figsize=(11, 6))
plotted_any = False

for label in ["Model 1 (baseline t-copula)", "Model 2 (Adverse contagion)", "Model 3 (Stress regime)"]:
    if label not in resolved:
        continue
    suf = resolved[label]

    cols = [f"EL_sen_{pp:02d}_{suf}" for pp in _s_levels]
    have = [c for c in cols if c in df_all.columns]
    if len(have) < 3:
        print(f"[warn] Not enough EL_sen_* columns for '{label}' (suffix '{suf}'). Found: {have}")
        continue

    M = df_all[have].mean(axis=0).to_numpy(dtype=float)
    xs = [int(c.split("_")[2]) / 100.0 for c in have]
    order = np.argsort(xs)
    xs = np.array(xs)[order]
    M  = M[order] * 100.0  # convert to %

    ls, lw = line_styles.get(label, ("-", 1.6))
    plt.plot(xs, M, linestyle=ls, linewidth=lw, label=label)
    plotted_any = True

# Optional horizontal reference: German Bund average
if plotted_any and "EL_germany_MC" in df_all.columns:
    bund_avg_pct = float(df_all["EL_germany_MC"].mean() * 100.0)
    plt.axhline(bund_avg_pct, ls="--", lw=1.0, color="black", alpha=0.7, label="German Bund")

# ----------------------------- Axes formatting -----------------------------
ax = plt.gca()
ax.set_xlabel("Subordination Level", fontsize=12)
ax.set_ylabel("Senior EL (%)", fontsize=12)
ax.set_title("Senior Expected Loss vs Subordination — mean across months", fontsize=12)

ax.set_xticks([p/100 for p in _s_levels])
ax.set_xticklabels([f"{p}%" for p in _s_levels])
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(round(y))}%"))

# Ticks (major only)
ax.tick_params(axis="x", which="major", direction="out", length=4, width=0.7, bottom=True, top=False)
ax.tick_params(axis="y", which="major", direction="out", length=4, width=0.7, left=True, right=False)
ax.tick_params(axis="both", which="minor", bottom=False, left=False)

# Grid and frame
ax.yaxis.grid(True)
ax.xaxis.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend with thin border
leg = ax.legend(loc="upper right", frameon=True, fontsize=10, facecolor="white", edgecolor="black")
leg.get_frame().set_linewidth(0.8)

plt.tight_layout()
plt.show()

# %% fig 18
# Safe-Asset Multiplier (Optimal s*) — apply existing style, just customize this axes

fig, ax = plt.subplots(figsize=(11, 5.2))

# Legend names per your mapping
labels = {
    "safe_asset_multiplier_optimal_Copula": "Model 1 (baseline t-copula)",
    "safe_asset_multiplier_optimal_AdvCon": "Model 2 (Adverse contagion)",
    "safe_asset_multiplier_optimal_Stress": "Model 3 (Stress regime)",
}

# Line styles to distinguish models (monochrome)
linestyles = {
    "safe_asset_multiplier_optimal_Copula": "-",
    "safe_asset_multiplier_optimal_AdvCon": "--",
    "safe_asset_multiplier_optimal_Stress": ":",
}

for col in ["safe_asset_multiplier_optimal_Copula",
            "safe_asset_multiplier_optimal_AdvCon",
            "safe_asset_multiplier_optimal_Stress"]:
    if col in df_all.columns:
        ax.plot(
            df_all.index,
            df_all[col],
            label=labels[col],
            linewidth=1.6,
            linestyle=linestyles[col],
        )

# Labels & title
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Safe Asset Multiplier", fontsize=12)
ax.set_title("Safe-asset multiplier — optimal subordination $s^*$", fontsize=12)

# Grid (horizontal only) & frame
ax.yaxis.grid(True)
ax.xaxis.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Ticks: clean major marks only
ax.tick_params(axis="x", which="major", direction="out", length=4, width=0.7, bottom=True, top=False, labelsize=12)
ax.tick_params(axis="y", which="major", direction="out", length=4, width=0.7, left=True, right=False, labelsize=12)
ax.tick_params(axis="both", which="minor", bottom=False, left=False)

# Legend inside with thin border
leg = ax.legend(loc="upper left", frameon=True, fontsize=10, facecolor="white", edgecolor="black")
leg.get_frame().set_linewidth(0.8)

# Use your existing date formatter, if available
try:
    pretty_date_axis(ax)
except Exception:
    pass

fig.tight_layout()
plt.savefig("FinalFigures/SafeAssetMultiplier_optimal_CopulaAdvConStress.png", dpi=300, bbox_inches="tight")
plt.show()

# %% FIG 19
# Optimal Subordination s* — apply existing style, customize this axes only
# Optimal Subordination s* — academic style with y-axis in %
import matplotlib.ticker as mtick

fig, ax = plt.subplots(figsize=(11, 5.2))

# Legend names mapping
labels = {
    "optimal_sub_Copula": "Model 1 (baseline t-copula)",
    "optimal_sub_AdvCon": "Model 2 (Adverse contagion)",
    "optimal_sub_Stress": "Model 3 (Stress regime)",
}

# Monochrome line styles
linestyles = {
    "optimal_sub_Copula": "-",
    "optimal_sub_AdvCon": "--",
    "optimal_sub_Stress": ":",
}

for col in ["optimal_sub_Copula", "optimal_sub_AdvCon", "optimal_sub_Stress"]:
    if col in df_all.columns:
        ax.plot(
            df_all.index,
            df_all[col] * 100,  # convert to percentage
            label=labels[col],
            linewidth=1.6,
            linestyle=linestyles[col],
        )

# Labels & title
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Subordination Level (%)", fontsize=12)
ax.set_title("Optimal Subordination $s^*$ — Dependence, Contagion & Stress Raise Protection", fontsize=12)

# Grid (horizontal only) & frame
ax.yaxis.grid(True)
ax.xaxis.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Tick formatting and appearance
ax.yaxis.set_major_formatter(mtick.PercentFormatter(100, decimals=0))
ax.tick_params(axis="x", which="major", direction="out", length=4, width=0.7, bottom=True, top=False, labelsize=12)
ax.tick_params(axis="y", which="major", direction="out", length=4, width=0.7, left=True, right=False, labelsize=12)
ax.tick_params(axis="both", which="minor", bottom=False, left=False)

# Legend inside with thin border
leg = ax.legend(loc="upper right", frameon=True, fontsize=10, facecolor="white", edgecolor="black")
leg.get_frame().set_linewidth(0.8)

# Use your existing date formatter if defined
try:
    pretty_date_axis(ax)
except Exception:
    pass

fig.tight_layout()
plt.savefig("FinalFigures/OptimalSubordination_CopulaAdvConStress.png", dpi=300, bbox_inches="tight")
plt.show()


# %%FIG 20
# %% SETUP: fonts, styling, imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick
from matplotlib.ticker import PercentFormatter
import seaborn as sns

# Global matplotlib style — monochrome academic
mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.color": "0.85",
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
    "axes.grid.axis": "y",                 # horizontal gridlines only
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "CMU Serif Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
})

# ================================================================
# FIG: Optimal Subordination s* — one y tick per labeled value
# ================================================================
fig, ax = plt.subplots(figsize=(11, 5.2))

# Legend names mapping
labels = {
    "optimal_sub_Copula": "Model 1 (baseline t-copula)",
    "optimal_sub_AdvCon": "Model 2 (Adverse contagion)",
    "optimal_sub_Stress": "Model 3 (Stress regime)",
}

# Monochrome line styles
linestyles = {
    "optimal_sub_Copula": "-",
    "optimal_sub_AdvCon": "--",
    "optimal_sub_Stress": ":",
}

# Plot series (percent units on y)
for col in ["optimal_sub_Copula", "optimal_sub_AdvCon", "optimal_sub_Stress"]:
    if col in df_all.columns:
        ax.plot(
            df_all.index,
            df_all[col] * 100.0,             # convert to percentage points
            label=labels[col],
            linewidth=1.6,
            linestyle=linestyles[col],
            color="black"
        )

# Labels (serif, 12 pt)
ax.set_xlabel("Years", fontsize=12, color="black")
ax.set_ylabel("Subordination Level (%)", fontsize=12, color="black")

# Optional compact title (<=12 pt to stay subtle)
ax.set_title("Optimal Subordination $s^*$ — Dependence, Contagion & Stress Raise Protection", fontsize=12, color="black")

# Axes frame (no top/right)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ---- Y-axis: single set of ticks (no minors), range to 35% ----
ax.set_ylim(0, 35)                                           # 0–35 percentage points
ax.yaxis.set_major_locator(mtick.MultipleLocator(5))         # labeled every 5%
ax.yaxis.set_minor_locator(mtick.NullLocator())              # <- remove minor ticks entirely
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))

# X ticks
ax.tick_params(axis="x", which="major", length=4, width=0.7, labelsize=12, bottom=True, top=False)
ax.tick_params(axis="y", which="major", length=4, width=0.7, labelsize=12, left=True, right=False)
ax.tick_params(axis="both", which="minor", length=0)         # ensure no minor marks

# Legend inside with thin border
leg = ax.legend(loc="upper right", frameon=True, fontsize=10, facecolor="white", edgecolor="black")
leg.get_frame().set_linewidth(0.8)

# Keep your existing date formatter if present
try:
    pretty_date_axis(ax)
except Exception:
    pass

fig.tight_layout()
plt.savefig("FinalFigures/OptimalSubordination_CopulaAdvConStress.png", dpi=300, bbox_inches="tight")
plt.show()


# %% FIG 21
# SETUP: fonts, styling, imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick
from matplotlib.ticker import PercentFormatter
import seaborn as sns
import string

# Global matplotlib style — monochrome academic
mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.color": "0.85",
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
    "axes.grid.axis": "y",                 # horizontal gridlines only
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "CMU Serif Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
})

# ================================================================
# FIG: Optimal Subordination s* — one y tick set, cap at 40%
# ================================================================
fig, ax = plt.subplots(figsize=(11, 5.2))

# Legend names mapping
labels = {
    "optimal_sub_Copula": "Model 1 (baseline t-copula)",
    "optimal_sub_AdvCon": "Model 2 (Adverse contagion)",
    "optimal_sub_Stress": "Model 3 (Stress regime)",
}

# Monochrome line styles
linestyles = {
    "optimal_sub_Copula": "-",
    "optimal_sub_AdvCon": "--",
    "optimal_sub_Stress": ":",
}

# Plot series (percent units on y)
for col in ["optimal_sub_Copula", "optimal_sub_AdvCon", "optimal_sub_Stress"]:
    if col in df_all.columns:
        ax.plot(
            df_all.index,
            df_all[col] * 100.0,             # percentage points
            label=labels[col],
            linewidth=1.6,
            linestyle=linestyles[col],
            color="black"
        )

# Labels (serif, 12 pt)
ax.set_xlabel("Year", fontsize=12, color="black")
ax.set_ylabel("Subordination Level (%)", fontsize=12, color="black")

# Subtle title
ax.set_title("Optimal Subordination $s^*$ — Dependence, Contagion & Stress Raise Protection", fontsize=12, color="black")

# Axes frame (no top/right)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ---- Y-axis: 0–40%, single major ticks every 5%, no minors ----
ax.set_ylim(0, 40)
ax.yaxis.set_major_locator(mtick.MultipleLocator(5))
ax.yaxis.set_minor_locator(mtick.NullLocator())
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, decimals=0))

# X ticks
ax.tick_params(axis="x", which="major", length=4, width=0.7, labelsize=12, bottom=True, top=False)
ax.tick_params(axis="y", which="major", length=4, width=0.7, labelsize=12, left=True, right=False)
ax.tick_params(axis="both", which="minor", length=0)

# Legend inside with thin border
leg = ax.legend(loc="upper right", frameon=True, fontsize=10, facecolor="white", edgecolor="black")
leg.get_frame().set_linewidth(0.8)

# Keep your existing date formatter if present
try:
    pretty_date_axis(ax)
except Exception:
    pass

fig.tight_layout()
plt.savefig("FinalFigures/OptimalSubordination_CopulaAdvConStress.png", dpi=300, bbox_inches="tight")
plt.show()


# %% FIG 20

# Expected Loss by Tranche — academic B&W style (Models 1–3)
# Panel titles "Panel A - s=..", "Panel B - s=..", etc.; y up to 40%
# X-axis simplified to "Model 1", "Model 2", "Model 3"

# -------------------------------
# Build tidy table from df_all / VERSIONS
# -------------------------------
subs_pp   = [10, 30, 50, 70]
subs_keys = [f"{pp:02d}" for pp in subs_pp]

def _safe_col(df, name):
    return name if name in df.columns else None

rows = []
for label, suf in VERSIONS.items():
    # Skip MC and λ=0.50 versions
    if label.strip().lower() == "mc" or "λ=0.50" in label:
        continue

    pool_col = _safe_col(df_all, f"EL_pool_{suf}")
    if not pool_col:
        continue

    for pp, key in zip(subs_pp, subs_keys):
        sen_col = _safe_col(df_all, f"EL_sen_{key}_{suf}")
        if not sen_col:
            continue

        s = pp / 100.0
        sen_norm = df_all[sen_col]
        pool     = df_all[pool_col]

        # Convert back to tranche-normalized quantities
        sen_pool = sen_norm * (1.0 - s)
        jun_pool = pool - sen_pool
        jun_norm = jun_pool / s

        rows.append({"Model": label, "Subordination": f"s={pp}%", "Tranche": "Senior", "EL": sen_norm.dropna().mean()})
        rows.append({"Model": label, "Subordination": f"s={pp}%", "Tranche": "Junior", "EL": jun_norm.dropna().mean()})

el_grid = pd.DataFrame(rows)

if el_grid.empty:
    print("[EL grid] No data found for requested subordination checkpoints.")
else:
    # -------------------------------
    # Robust label → Model name mapping
    # -------------------------------
    def _map_model_name(x: str) -> str | None:
        xl = (x or "").lower()
        if "copula" in xl and ("opt" in xl or "(opt" in xl or "optimized" in xl):
            return "Model 1"
        if ("advcon" in xl) or ("adverse" in xl) or ("contagion" in xl) or ("copula+contagion" in xl) or ("copula + contagion" in xl):
            return "Model 2"
        if ("stress" in xl) or ("crisis" in xl):
            return "Model 3"
        if x in ("Stress",):
            return "Model 3"
        if x in ("Copula+Contagion",):
            return "Model 2"
        return None

    el_grid["ModelMapped"] = el_grid["Model"].apply(_map_model_name)
    el_grid = el_grid.dropna(subset=["ModelMapped"])

    # Order models consistently and keep only present ones
    model_order = [m for m in ["Model 1", "Model 2", "Model 3"] if m in el_grid["ModelMapped"].unique()]
    tranche_order = ["Senior", "Junior"]
    sub_order = [f"s={pp}%" for pp in subs_pp if f"s={pp}%" in el_grid["Subordination"].unique()]

    # -------------------------------
    # Plot (monochrome academic style; uniform y-range up to 40%)
    # -------------------------------
    n_sub = len(sub_order)
    ncols = min(2, n_sub)
    nrows = int(np.ceil(n_sub / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.8 * nrows), squeeze=False)
    axes = axes.flatten()

    # Monochrome palette for Tranche (neutral grays)
    palette = {"Senior": "#7f7f7f", "Junior": "#bfbfbf"}

    for i, sub in enumerate(sub_order):
        ax = axes[i]
        data_i = el_grid[el_grid["Subordination"] == sub].copy()
        data_i["ModelMapped"] = pd.Categorical(data_i["ModelMapped"], categories=model_order, ordered=True)
        data_i["Tranche"] = pd.Categorical(data_i["Tranche"], categories=tranche_order, ordered=True)

        sns.barplot(
            data=data_i, x="ModelMapped", y="EL", hue="Tranche",
            order=model_order, hue_order=tranche_order,
            ax=ax, palette=palette, saturation=1.0, linewidth=0
        )

        # ---- Panel titles with letters A, B, C, ... ----
        panel_letter = string.ascii_uppercase[i]
        ax.set_title(f"Panel {panel_letter} - {sub}", fontsize=12, color="black")

        ax.set_xlabel("")
        ax.set_ylabel("Expected Loss (%)", fontsize=12, color="black")

        # ---- Y as percent with single major ticks; shared range 0–40% ----
        ax.set_ylim(0.0, 0.40)
        ax.yaxis.set_major_locator(mtick.MultipleLocator(0.05))
        ax.yaxis.set_minor_locator(mtick.NullLocator())
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

        # Academic grid & frame
        ax.xaxis.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.9)
        ax.spines["bottom"].set_linewidth(0.9)

        # Ticks — thin, outward (no minors)
        ax.tick_params(axis="x", which="major", direction="out", length=4, width=0.7, labelsize=11)
        ax.tick_params(axis="y", which="major", direction="out", length=4, width=0.7, labelsize=11, left=True, right=False)
        ax.tick_params(axis="both", which="minor", length=0)

        for tick in ax.get_xticklabels():
            tick.set_rotation(0)

        # Legend with thin border inside
        leg = ax.legend(loc="upper right", frameon=True, fontsize=10, facecolor="white", edgecolor="black", title=None)
        leg.get_frame().set_linewidth(0.8)

    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig("FinalFigures/ExpectedLoss_by_Tranche_Models1to3.png", dpi=300, bbox_inches="tight")    
    plt.show()

# %%FIG 21
# Academic style with single-grey crisis shading + dotted outline and labels
# Updates:
# - Crisis #3 label aligned to the RIGHT edge of its box ("Energy crisis")
# - Legend placed to the RIGHT of the axes
# - Y-axis label text changed to "Years"
# - Keep Computer Modern serif, dotted horizontal grid, black model lines, explicit tick marks

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick
from matplotlib.patches import Rectangle, Patch
from matplotlib import dates as mdates
from matplotlib import transforms as mtransforms

# -----------------------------
# Global matplotlib style (Computer Modern, monochrome)
# -----------------------------
mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.color": "0.85",
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
    "axes.grid.axis": "y",                 # horizontal gridlines only
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 5,                 # emphasized tick marks
    "ytick.major.size": 5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "CMU Serif Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
})

# -----------------------------
# Shade crises: single grey fill + dotted outline + label per window
# -----------------------------
def shade_custom_crises(ax):
    """
    Shades crisis windows using a single light grey, adds a dotted black outline and
    places the crisis name above each box. For the third crisis, label is RIGHT-aligned
    at the end of the box with text 'Energy crisis'.
    Returns a single legend handle 'Crisis'.
    """
    spans = [
        ("Euro Area Debt Crisis", pd.Timestamp("2010-01-01"), pd.Timestamp("2013-12-31")),
        ("COVID-19 Shock",        pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31")),
        ("Energy crisis       ",         pd.Timestamp("2021-01-01"), pd.Timestamp("2023-12-31")),  # renamed
    ]
    grey_fill = "0.90"  # single shade for all crises

    # blended transform: x in data coords (dates), y in axes (0..1)
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    for idx, (lab, start, end) in enumerate(spans, start=1):
        # 1) Fill band in light grey
        ax.axvspan(start, end, color=grey_fill, alpha=1.0, zorder=0)

        # 2) Dotted rectangle outline spanning full axis height (0..1)
        x0 = mdates.date2num(start)
        x1 = mdates.date2num(end)
        width = x1 - x0
        rect = Rectangle((x0, 0.0), width, 1.0,
                         transform=trans,
                         facecolor="none",
                         edgecolor="black",
                         linestyle=":",
                         linewidth=0.8,
                         zorder=1)
        ax.add_patch(rect)

        # 3) Crisis label above the box
        if idx == 3:
            # For crisis #3: RIGHT-aligned at the end of the box
            ax.text(end, 1.02, lab, ha="right", va="bottom", fontsize=10,
                    transform=ax.get_xaxis_transform(), color="black")
        else:
            # Others centered
            x_mid = start + (end - start) / 2
            ax.text(x_mid, 1.02, lab, ha="center", va="bottom", fontsize=10,
                    transform=ax.get_xaxis_transform(), color="black")

    # Single legend entry for all shaded regions
    crisis_handle = Patch(facecolor=grey_fill, edgecolor="black", linewidth=0.8, label="Crisis")
    return [crisis_handle]

# -----------------------------
# Models & linestyles (monochrome)
# -----------------------------
models = [
    ("Copula", "Model 1"),
    ("AdvCon", "Model 2"),
    ("Stress", "Model 3"),
]
linestyles = {
    "Copula": "-",
    "AdvCon": "--",
    "Stress": ":",
}

# -----------------------------
# Figure
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 5.2))

plotted = False
for suf, lab in models:
    col = f"EL_pool_{suf}"
    if col in df_all.columns:
        ax.plot(
            df_all.index,
            df_all[col],                    # fractions (0..1)
            label=lab,
            linewidth=1.6,
            linestyle=linestyles.get(suf, "-"),
            color="black",
        )
        plotted = True

# Shade crisis windows and collect single legend handle
crisis_handles = shade_custom_crises(ax)

# Percent y-axis: single major ticks (every 5%), no minors
ax.yaxis.set_major_locator(mtick.MultipleLocator(0.05))
ax.yaxis.set_minor_locator(mtick.NullLocator())
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))

# Labels (y label text per request)
ax.set_ylabel("Pool Expected Loss (%)", fontsize=12, color="black")
ax.set_xlabel("Year", fontsize=12, color="black")

# Minimal frame (no top/right)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.9)
ax.spines["bottom"].set_linewidth(0.9)
ax.xaxis.grid(False)

# ----- Explicit tick marks on both axes -----
ax.tick_params(axis="x", which="major", direction="out", length=5, width=0.8, bottom=True, top=False, labelsize=12)
ax.tick_params(axis="y", which="major", direction="out", length=5, width=0.8, left=True, right=False, labelsize=12)

# Combined legend (models + single 'Crisis'), placed to the RIGHT of the axes
if plotted:
    model_handles, model_labels = ax.get_legend_handles_labels()
    leg = ax.legend(
        model_handles + crisis_handles,
        model_labels + [crisis_handles[0].get_label()],
        loc="upper right",
        borderaxespad=0.0,
        ncol=1,
        frameon=True,
        fontsize=10,
        facecolor="white",
        edgecolor="black",
    )
    leg.get_frame().set_linewidth(0.8)

# Optional: pretty date formatting if available
try:
    pretty_date_axis(ax)
except Exception:
    pass

# Make room for the right-side legend
plt.subplots_adjust(right=0.80)
fig.tight_layout()
plt.savefig("FinalFigures/PoolEL_CopulaAdvConStress_withCrisisShading.png", dpi=300, bbox_inches="tight")
plt.show()



# %%  FIG 22, 23, 24
# Academic finance style — Safe-Asset Multiplier (monochrome, serif, dotted horizontal grid)
# Update: Graph 3 uses ticks every 4 months (abbr month + short year), e.g., "Feb '21"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pandas.tseries.offsets import MonthEnd
import matplotlib.dates as mdates
from matplotlib import ticker as mticker  # for NullLocator

# -----------------------------
# Global Matplotlib style (Computer Modern, monochrome)
# -----------------------------
mpl.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "axes.grid.axis": "y",        # horizontal gridlines only
    "grid.color": "0.85",
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "CMU Serif Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
})

# -----------------------------
# Model labels (suffix, pretty)
# -----------------------------
MODELS = [
    ("Copula", "Model 1"),
    ("AdvCon", "Model 2"),
    ("Stress", "Model 3"),
]

# Metric columns we care about
METRIC_OPT = "safe_asset_multiplier_optimal"
METRIC_S30 = "safe_asset_multiplier_30"

def _window_bounds(start_str, end_str):
    """Return inclusive [start, end] snapped to month-ends."""
    s = pd.Timestamp(start_str)
    e = pd.Timestamp(end_str)
    s_me = s + MonthEnd(0)  # snap to month-end
    e_me = e + MonthEnd(0)  # inclusive upper bound at month-end
    return s_me.normalize(), e_me.normalize()

def _style_axes(ax, xlabel_text):
    """Apply shared axes styling."""
    ax.set_xlabel(xlabel_text, fontsize=12, color="black")
    ax.set_ylabel("Safe Asset Multiplier", fontsize=12, color="black")
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="x", which="major", direction="out", length=4, width=0.7, labelsize=12, bottom=True, top=False)
    ax.tick_params(axis="y", which="major", direction="out", length=4, width=0.7, labelsize=12, left=True, right=False)

def _plot_series(ax, df_all, lo, hi):
    """Plot the four requested series with the specified styles."""
    # OPTIMAL lines (three models)
    for suf, model_label in MODELS:
        col_opt = f"{METRIC_OPT}_{suf}"
        if col_opt not in df_all.columns:
            print(f"[warn] Missing column: {col_opt}")
            continue
        g = df_all[[col_opt]].copy().loc[lo:hi].dropna()
        g.columns = ["Value"]
        g["Month"] = g.index
        if model_label == "Model 1":
            ls = "-"
            color = "black"
            label = "Model 1 (optimal)"
        elif model_label == "Model 2":
            ls = "--"
            color = "black"
            label = "Model 2 (optimal)"
        else:
            ls = ":"
            color = "black"
            label = "Model 3 (optimal)"
        ax.plot(g["Month"], g["Value"], linewidth=1.6, linestyle=ls, color=color, label=label)

    # Model 1 at s = 30% — dark grey, long dashed (shortened ~2/3)
    col_s30_m1 = f"{METRIC_S30}_Copula"
    if col_s30_m1 in df_all.columns:
        g = df_all[[col_s30_m1]].copy().loc[lo:hi].dropna()
        g.columns = ["Value"]
        g["Month"] = g.index
        line_m1_s30, = ax.plot(
            g["Month"], g["Value"],
            linewidth=1.6,
            linestyle="-",  # override with custom pattern below
            color="0.25",
            label="30% subordination",
        )
        line_m1_s30.set_linestyle((0, (10, 4)))  # custom long-dash

def _legend(ax):
    """Place legend slightly above middle-right."""
    leg = ax.legend(
        title="Models",
        loc="center right",
        bbox_to_anchor=(1.0, 0.62),  # a bit above center
        frameon=True,
        fontsize=10,
        facecolor="white",
        edgecolor="black",
    )
    if leg.get_title() is not None:
        leg.get_title().set_fontsize(10)
    leg.get_frame().set_linewidth(0.8)

def _graph(df_all, start, end, title_suffix, xlabel_text, tick_mode):
    """
    tick_mode:
      - 'years_only': only January ticks with year labels
      - 'every_2_months_abbrev_shortyear': every 2 months, labels like "Feb '21"
      - 'every_4_months_abbrev_shortyear': every 4 months, labels like "Feb '21"
    """
    lo, hi = _window_bounds(start, end)
    fig, ax = plt.subplots(figsize=(12, 5.6))

    _plot_series(ax, df_all, lo, hi)

    if title_suffix:
        ax.set_title(f"Safe-Asset Multiplier — {title_suffix}", fontsize=12, color="black")

    _style_axes(ax, xlabel_text)
    ax.set_xlim(lo, hi)

    # Apply requested tick scheme
    if tick_mode == "years_only":
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=1, interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
    elif tick_mode == "every_2_months_abbrev_shortyear":
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
    elif tick_mode == "every_4_months_abbrev_shortyear":
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_minor_locator(mticker.NullLocator())

    _legend(ax)

    fig.tight_layout()
    plt.savefig(f"FinalFigures/SafeAssetMultiplier_{title_suffix.replace(' ', '_').replace('–', '-')}.png", dpi=300, bbox_inches="tight")   
    plt.show()

# ----- Graph 1: Jan 2010 to Dec 2014 — only January ticks (years) -----
_graph(
    df_all,
    start="2010-01-01",
    end="2014-12-31",
    title_suffix="Jan 2010 – Dec 2014",
    xlabel_text="Year",
    tick_mode="years_only",
)

# ----- Graph 2: Mar 2020 to Dec 2021 — abbreviated months every 2 months ("Feb '21" style) -----
_graph(
    df_all,
    start="2020-02-01",
    end="2021-12-31",
    title_suffix="Mar 2020 – Dec 2021",
    xlabel_text="Month",
    tick_mode="every_2_months_abbrev_shortyear",
)

# ----- Graph 3: Jan 2021 to Dec 2023 — abbreviated months every 4 months ("Feb '21" style) -----
_graph(
    df_all,
    start="2021-01-01",
    end="2023-12-31",
    title_suffix="Jan 2021 – Dec 2023",
    xlabel_text="Month",
    tick_mode="every_4_months_abbrev_shortyear",
)

# %%
