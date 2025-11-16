#%% Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns  

#%%
# Monte Carlo results (t-copula variants only)
# Version 2 with t-copula, optimized lambda and fixed lambda (0.50)
df_copu = pd.read_csv("data /MC_results_tcopula.csv", index_col=0)
df_copu_fixedlambda = pd.read_csv("data /MC_results_tcopula_lambda050.csv", index_col=0)
# Version 3 with t-copula and adverse contagion
df_adv  = pd.read_csv("data /MC_results_tcopula_adverse.csv", index_col=0)
# Version 4 with t-copula and crisis regime (stress test)
df_stress  = pd.read_csv("data /MC_results_tcopula_stress.csv", index_col=0)

#%% Ensure datetime index (months) for plotting
for df in [df_copu, df_copu_fixedlambda, df_adv, df_stress]:
    # Try to coerce to datetime if not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

#%% Rename columns with suffixes
df_copu   = df_copu.add_suffix("_Copula")
df_copu_fixedlambda = df_copu_fixedlambda.add_suffix("_Copula050")
df_adv    = df_adv.add_suffix("_AdvCon")
df_stress = df_stress.add_suffix("_Stress")

#%% Merge on Month index
df_all = pd.concat([df_copu, df_copu_fixedlambda, df_adv, df_stress], axis=1)
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
plt.plot(df_all.index, df_all.get("optimal_sub_Copula"),     label="Copula (opt λ)", linewidth=1.6)
plt.plot(df_all.index, df_all.get("optimal_sub_Copula050"),  label="Copula (λ=0.50)", linewidth=1.6)
plt.plot(df_all.index, df_all.get("optimal_sub_AdvCon"),     label="Copula + Contagion", linewidth=1.6)
plt.plot(df_all.index, df_all.get("optimal_sub_Stress"),     label="Crisis Regime (Stress)", linewidth=1.6)
plt.title("Optimal Subordination $s^*$ — Dependence, Contagion & Stress Raise Protection")
plt.ylabel("Subordination (fraction)"); plt.xlabel("Month")
plt.grid(True, alpha=0.3); plt.legend()
pretty_date_axis(plt.gca())
plt.tight_layout(); plt.show()

#%% ---------- Figure 2: Senior EL at s = 0.30 ----------
plt.figure(figsize=(11,5.2))
plt.plot(df_all.index, df_all.get("EL_sen_30_Copula"),     label="Copula (opt λ)", linewidth=1.6)
plt.plot(df_all.index, df_all.get("EL_sen_30_Copula050"),  label="Copula (λ=0.50)", linewidth=1.6)
plt.plot(df_all.index, df_all.get("EL_sen_30_AdvCon"),     label="Copula + Contagion", linewidth=1.6)
plt.plot(df_all.index, df_all.get("EL_sen_30_Stress"),     label="Crisis Regime (Stress)", linewidth=1.6)
plt.title("Senior EL at Subordination = 0.30 — All Versions")
plt.ylabel("Expected Loss"); plt.xlabel("Month")
plt.grid(True, alpha=0.3); plt.legend()
pretty_date_axis(plt.gca())
plt.tight_layout(); plt.show()

#%% ---------- Figure 3: Safe-asset multipliers (optimal and s = 0.30) ----------
fig, ax = plt.subplots(2,1, figsize=(11,8), sharex=True)

# Optimal s*
ax[0].plot(df_all.index, df_all.get("safe_asset_multiplier_optimal_Copula"),     label="Copula (opt λ)", linewidth=1.6)
ax[0].plot(df_all.index, df_all.get("safe_asset_multiplier_optimal_Copula050"),  label="Copula (λ=0.50)", linewidth=1.6)
ax[0].plot(df_all.index, df_all.get("safe_asset_multiplier_optimal_AdvCon"),     label="Copula + Contagion", linewidth=1.6)
ax[0].plot(df_all.index, df_all.get("safe_asset_multiplier_optimal_Stress"),     label="Crisis Regime (Stress)", linewidth=1.6)
ax[0].set_title("Safe-Asset Multiplier (Optimal $s^*$) — Output Safe Assets per € of Bund input")
ax[0].set_ylabel("Multiplier"); ax[0].grid(True, alpha=0.3); ax[0].legend()

# Fixed s = 0.30
ax[1].plot(df_all.index, df_all.get("safe_asset_multiplier_30_Copula"),     label="Copula (opt λ)", linewidth=1.6)
ax[1].plot(df_all.index, df_all.get("safe_asset_multiplier_30_Copula050"),  label="Copula (λ=0.50)", linewidth=1.6)
ax[1].plot(df_all.index, df_all.get("safe_asset_multiplier_30_AdvCon"),     label="Copula + Contagion", linewidth=1.6)
ax[1].plot(df_all.index, df_all.get("safe_asset_multiplier_30_Stress"),     label="Crisis Regime (Stress)", linewidth=1.6)
ax[1].set_title("Safe-Asset Multiplier (Fixed s = 0.30)")
ax[1].set_ylabel("Multiplier"); ax[1].set_xlabel("Month"); ax[1].grid(True, alpha=0.3); ax[1].legend()

pretty_date_axis(ax[1])
plt.tight_layout(); plt.show()

#%% ---------- Figure 5: Histograms — EL_jun_optimal (tail risk) ----------
plt.figure(figsize=(11,5.2))
hist_specs = [
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
    "Copula":     df_all.get("EL_sen_30_Copula"),
    "Copula050":  df_all.get("EL_sen_30_Copula050"),
    "AdvCon":     df_all.get("EL_sen_30_AdvCon"),
    "Stress":     df_all.get("EL_sen_30_Stress"),
}).rolling(roll).std()
plt.figure(figsize=(11,5.2))
for c, lab in [("Copula","Copula (opt λ)"),
               ("Copula050","Copula (λ=0.50)"),
               ("AdvCon","Copula + Contagion"),
               ("Stress","Crisis Regime (Stress)")]:
    if c in vol.columns:
        plt.plot(vol.index, vol[c], label=lab, linewidth=1.6)
plt.title(f"Rolling {roll}-Month Volatility of Senior EL @ 0.30 — All Versions")
plt.ylabel("Rolling Std. Dev."); plt.xlabel("Month")
plt.grid(True, alpha=0.3); plt.legend()
pretty_date_axis(plt.gca())
plt.tight_layout(); plt.show()

#%% ---------- Figure 7: Pool99 comparison ----------
plt.figure(figsize=(11,5.2))
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

# (rolling correlation vs MC block removed, as it depended on MC baseline)

#===============================================================
# 8) “WHICH VERSION IS LOWEST?” — head-to-head frequency
#    For each month, which version yields the *lowest* EL_sen_30?
#===============================================================
"""
metric = "EL_sen_30"
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

"""
metric = "safe_asset_multiplier_optimal"
plt.figure(figsize=(11,5.2))
for label, suf in VERSIONS.items():
    c = col(metric, suf)
    if c:
        dd = compute_drawdown(df_all[c])
        plt.plot(dd.index, dd, label=label, alpha=0.9)
plt.title("Drawdowns — Safe-Asset Multiplier (Optimal $s^*$)")
plt.ylabel("Drawdown"); plt.xlabel("Month")
plt.grid(True, alpha=0.3); plt.legend()
pretty_date_axis(plt.gca())
plt.tight_layout(); plt.show()
"""

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
# (QQ, scatter, hist-diff, KS summaries left commented as in your original)
#-----------------------------
"""
def qqplot(a, b, label_a, label_b, metric):
    ...
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

# (hist of differences & KS stats block kept commented)

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

# (heatmaps & ECDF for junior kept commented as in your script)

#%%
# --- styling ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("talk")
sns.set_style("whitegrid")

# Subordination checkpoints you saved
_s_levels = [0, 10, 20, 30, 40, 50, 60, 70]              # as percentages
_s_cols   = [f"{pp:02d}" for pp in _s_levels]            # "00","10",...

# ---- choose a baseline version (here: Copula opt λ) ----
baseline_keys = []
for label, suf in VERSIONS.items():
    ll = label.lower()
    if ("copula" in ll and "opt" in ll) or suf == "Copula":
        baseline_keys.append((label, suf))

if not baseline_keys:
    raise ValueError("Couldn't find a Copula baseline in VERSIONS. "
                     "Make sure VERSIONS contains something like {'Copula (opt λ)':'Copula'}.")

label, suf = baseline_keys[0]  # take the first match

# --------------------- Plot: EL(s) curve (time-avg) for baseline ---------------------
#%%
plt.figure(figsize=(11, 6))

cols = [f"EL_sen_{pp:02d}_{suf}" for pp in _s_levels]
have = [c for c in cols if c in df_all.columns]
if len(have) < 3:
    raise ValueError(f"Not enough EL_sen_* columns for suffix '{suf}'. Found: {have}")

# Time-aggregate: mean across months
M  = df_all[have].mean(axis=0).to_numpy()

# x-axis from the column names we have (keeps alignment even if some are missing)
xs = [int(c.split("_")[2]) / 100.0 for c in have]
order = np.argsort(xs)
xs = np.array(xs)[order]; M = M[order]

plt.plot(xs, M, marker="o", linewidth=2, label=label)

# AAA threshold at 0.5% EL
AAA_EL = 0.005  # 0.5%
plt.axhline(AAA_EL, ls=":", alpha=0.9, linewidth=2, label="AAA threshold (0.5% EL)")

# ---- cosmetics ----
plt.title("Senior Expected Loss vs Subordination — Copula baseline (mean across months)")
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
Q1 = df_all[have].quantile(0.25, axis=0).to_numpy()
Q3 = df_all[have].quantile(0.75, axis=0).to_numpy()

# x from the column names we actually have (keeps order correct)
xs = [int(c.split("_")[2]) / 100.0 for c in have]
order = np.argsort(xs)
xs = np.array(xs)[order]; M = M[order]; Q1 = Q1[order]; Q3 = Q3[order]

# plot mean + IQR
plt.plot(xs, M, marker="o", linewidth=2.2, label=f"{label_cop}")
plt.fill_between(xs, Q1, Q3, alpha=0.20, linewidth=0)

# horizontal reference: AAA threshold
AAA_EL = 0.005  # 0.5%
plt.axhline(AAA_EL, ls=":", alpha=0.9, linewidth=2, label="AAA threshold (0.5% EL)")

# cosmetics
plt.title("Senior Expected Loss vs Subordination — t-Copula (opt λ) (mean across months)")
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
    # Skip λ=0.50 version
    if "λ=0.50" in label:
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
    # Ensure consistent order across facets
    model_order = [m for m in el_grid["Model"].unique()]
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
    s_me = (s + MonthEnd(0)) if s.day != 1 else (s + MonthEnd(0))
    e_me = e + MonthEnd(0)
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
    ax.set_xlim(lo, hi)
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

# ----- Figure A: Jan 2010 to Dec 2014 -----
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

# Force ticks
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

# Academic-style Safe-Asset Multiplier plot (Fixed s = 0.30)
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

# No minor ticks
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

import re
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.ticker import FuncFormatter

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

target_series = {
    "Model 1 (baseline t-copula)": ["Copula"],
    "Model 2 (Adverse contagion)": ["AdvCon", "Adverse", "Contagion", "Adv"],
    "Model 3 (Stress regime)": ["Stress", "Crisis", "StressTest", "RegimeStress"],
}

resolved = {}
for label, candidates in target_series.items():
    suf = _pick_suffix(df_all, candidates)
    if suf is not None:
        resolved[label] = suf
    else:
        print(f"[warn] No suffix found for '{label}' among candidates {candidates}. Skipping this line.")

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
    M  = M[order] * 100.0

    ls, lw = line_styles.get(label, ("-", 1.6))
    plt.plot(xs, M, linestyle=ls, linewidth=lw, label=label)
    plotted_any = True

ax = plt.gca()
ax.set_xlabel("Subordination Level", fontsize=12)
ax.set_ylabel("Senior EL (%)", fontsize=12)
ax.set_title("Senior Expected Loss vs Subordination — mean across months", fontsize=12)

ax.set_xticks([p/100 for p in _s_levels])
ax.set_xticklabels([f"{p}%" for p in _s_levels])
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(round(y))}%"))

ax.tick_params(axis="x", which="major", direction="out", length=4, width=0.7, bottom=True, top=False)
ax.tick_params(axis="y", which="major", direction="out", length=4, width=0.7, left=True, right=False)
ax.tick_params(axis="both", which="minor", bottom=False, left=False)

ax.yaxis.grid(True)
ax.xaxis.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

leg = ax.legend(loc="upper right", frameon=True, fontsize=10, facecolor="white", edgecolor="black")
leg.get_frame().set_linewidth(0.8)

plt.tight_layout()
plt.show()

# %% fig 18
# Safe-Asset Multiplier (Optimal s*) — apply existing style, just customize this axes

fig, ax = plt.subplots(figsize=(11, 5.2))

labels = {
    "safe_asset_multiplier_optimal_Copula": "Model 1 (baseline t-copula)",
    "safe_asset_multiplier_optimal_AdvCon": "Model 2 (Adverse contagion)",
    "safe_asset_multiplier_optimal_Stress": "Model 3 (Stress regime)",
}

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

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Safe Asset Multiplier", fontsize=12)
ax.set_title("Safe-asset multiplier — optimal subordination $s^*$", fontsize=12)

ax.yaxis.grid(True)
ax.xaxis.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.tick_params(axis="x", which="major", direction="out", length=4, width=0.7, bottom=True, top=False, labelsize=12)
ax.tick_params(axis="y", which="major", direction="out", length=4, width=0.7, left=True, right=False, labelsize=12)
ax.tick_params(axis="both", which="minor", bottom=False, left=False)

leg = ax.legend(loc="upper left", frameon=True, fontsize=10, facecolor="white", edgecolor="black")
leg.get_frame().set_linewidth(0.8)

try:
    pretty_date_axis(ax)
except Exception:
    pass

fig.tight_layout()
plt.savefig("FinalFigures/SafeAssetMultiplier_optimal_CopulaAdvConStress.png", dpi=300, bbox_inches="tight")
plt.show()

# %% FIG 19
# Optimal Subordination s* — academic style with y-axis in %
import matplotlib.ticker as mtick

fig, ax = plt.subplots(figsize=(11, 5.2))

labels = {
    "optimal_sub_Copula": "Model 1 (baseline t-copula)",
    "optimal_sub_AdvCon": "Model 2 (Adverse contagion)",
    "optimal_sub_Stress": "Model 3 (Stress regime)",
}

linestyles = {
    "optimal_sub_Copula": "-",
    "optimal_sub_AdvCon": "--",
    "optimal_sub_Stress": ":",
}

for col in ["optimal_sub_Copula", "optimal_sub_AdvCon", "optimal_sub_Stress"]:
    if col in df_all.columns:
        ax.plot(
            df_all.index,
            df_all[col] * 100,
            label=labels[col],
            linewidth=1.6,
            linestyle=linestyles[col],
        )

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Subordination Level (%)", fontsize=12)
ax.set_title("Optimal Subordination $s^*$ — Dependence, Contagion & Stress Raise Protection", fontsize=12)

ax.yaxis.grid(True)
ax.xaxis.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.yaxis.set_major_formatter(mtick.PercentFormatter(100, decimals=0))
ax.tick_params(axis="x", which="major", direction="out", length=4, width=0.7, bottom=True, top=False, labelsize=12)
ax.tick_params(axis="y", which="major", direction="out", length=4, width=0.7, left=True, right=False, labelsize=12)
ax.tick_params(axis="both", which="minor", bottom=False, left=False)

leg = ax.legend(loc="upper right", frameon=True, fontsize=10, facecolor="white", edgecolor="black")
leg.get_frame().set_linewidth(0.8)

try:
    pretty_date_axis(ax)
except Exception:
    pass

fig.tight_layout()
plt.savefig("FinalFigures/OptimalSubordination_CopulaAdvConStress.png", dpi=300, bbox_inches="tight")
plt.show()


# %%FIG 20
# (Pool expected loss with crisis shading — already only uses Copula/AdvCon/Stress in your code)
# ... [kept identical to your version, which has no df_norm/MC dependency] ...

# (And similarly for the final block with three Safe-Asset Multiplier windows – already Copula/AdvCon/Stress only)
