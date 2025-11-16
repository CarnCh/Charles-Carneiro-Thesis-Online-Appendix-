#%% --- imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------
# 0) Load your files
# -------------------
#%%
gdp = pd.read_csv("data /yearly_gdp.csv")
pd_weight = pd.read_csv("data /PDs+Weights.csv", index_col=0)
all_df = pd.read_csv("data /Merged_df.csv")
MAdf = pd.read_excel("data /Market_Access_2010_2025_with_NonEuroAreaPlayers.xlsx")
#%%
# ===========================================================
# Helper: make a datetime index from strings or a date column
# ============================================================
def ensure_datetime_index(df: pd.DataFrame):
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    # try common date/time columns, else first column
    candidates = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()] + [df.columns[0]]
    for c in candidates:
        try:
            dt = pd.to_datetime(df[c], errors="raise")
            out = df.copy()
            out.index = dt
            if c in out.columns: out = out.drop(columns=[c])
            return out.sort_index()
        except Exception:
            continue
    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    return out.sort_index()

#%% N°1
# ============================================
# 1) PDs across time for all countries
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Load and prepare data ---
pd_weight = pd.read_csv("data /PDs+Weights.csv", index_col=0)
pdw = pd_weight.copy()
pdw.index = pd.to_datetime(pdw.index, errors="coerce").to_period("M").to_timestamp("M")
pdw = pdw.sort_index()

# Choose PD columns
pd_cols = [c for c in pdw.columns if c.endswith("_PDs") or c.endswith("_PD")]
if not pd_cols:
    pd_cols = pdw.select_dtypes(include=[np.number]).columns.tolist()

def clean_label(label: str):
    for suf in ["_PDs", "_PD"]:
        if label.endswith(suf):
            label = label[:-len(suf)]
    return label.strip().replace("_", " ")

# --- Plot styling ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif"],
    "mathtext.fontset": "cm",
    "axes.edgecolor": "black",
    "axes.linewidth": 0.9,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10,
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
})

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

lw = 1.4

# --- Plot lines ---
n_countries = len(pd_cols)
for i, c in enumerate(pd_cols):
    linestyle = "-" if i < 10 else "--"  # first 9 solid, rest dashed
    ax.plot(
        pdw.index,
        pdw[c].astype(float),
        linewidth=lw,
        linestyle=linestyle,
        label=clean_label(c),
    )

# --- Axes and labels ---
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Probability of Default (%)", fontsize=12)
ax.set_ylim(0, 60)
ax.set_xlim(pdw.index.min(), pdw.index.max())
ax.margins(x=0)

# --- Grid and frame ---
ax.grid(True, axis="y", which="major", linestyle=":", linewidth=0.8)
ax.grid(False, axis="x")

for spine in ax.spines.values():
    spine.set_linewidth(0.9)
ax.tick_params(direction="out", length=3.0, width=0.8, pad=2)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y):d}%"))
ax.spines[["top", "right"]].set_visible(False)

# --- Legend ---
ax.legend(loc="best", ncol=1, frameon=True)

plt.tight_layout()
plt.savefig("FinalFigures/FIG_1.png", dpi=300)
plt.show()


#%% N°2 
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import os

# -------------------------------
# Minimal global style (B&W only)
# -------------------------------
plt.rcParams.update({
    # Typography
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif"],
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

    # Grid (horizontal only; subtle dotted)
    "grid.color": "0.85",
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,

    # Enforce monochrome lines; distinguish by linestyle
    "axes.prop_cycle": cycler("color", ["black"]),
})

# =========================================================
# ----- Italy GDP (requires DataFrames `gdp` and `all_df`)
# =========================================================

# Clean and reshape GDP (yearly_gdp.csv-like)
_drop = ['DATAFLOW','LAST UPDATE','freq','unit','na_item','OBS_FLAG','CONF_STATUS']
gdp = gdp.drop(columns=[c for c in _drop if c in gdp.columns], errors="ignore")
gdp_wide = gdp.pivot(index="TIME_PERIOD", columns="geo", values="OBS_VALUE")

# Find Italy column
cands = ["Italy", "ITALY", "IT", "ITA"]
g_col = next((c for c in cands if c in gdp_wide.columns),
             next((c for c in gdp_wide.columns if "ital" in str(c).lower()), None))
if g_col is None:
    raise KeyError("Italy not found in yearly_gdp.csv (looked for IT/ITA/Italy).")

# Year-end index
gdp_wide.index = pd.to_datetime(gdp_wide.index.astype(str), format="%Y") + pd.offsets.YearEnd(0)
italy_gdp_yearly = pd.to_numeric(gdp_wide[g_col], errors="coerce").rename("Italy GDP (Yearly)")

# Moving average series from Merged_df.csv-like `all_df`
all_dt = all_df.copy()
date_col = all_dt.columns[0]
all_dt.index = pd.to_datetime(all_dt[date_col], errors="coerce")
all_dt = all_dt.drop(columns=[date_col], errors="ignore").sort_index()

ma_col = next((c for c in all_dt.columns if c in ["Italy MA-5","ITALY MA-5","IT MA-5","ITA MA-5"]),
              next((c for c in all_dt.columns if ("ital" in c.lower()) and ("ma-5" in c.lower())), None))
if ma_col is None:
    raise KeyError("Column 'Italy MA-5' (or similar) not found in Merged_df.csv.")

italy_ma_monthly = pd.to_numeric(all_dt[ma_col], errors="coerce").rename("Italy GDP (MA-5)")
italy_ma_yearly = italy_ma_monthly.resample("Y").last()

# Apply start years
italy_gdp_yearly = italy_gdp_yearly[italy_gdp_yearly.index.year >= 2006]
italy_ma_yearly  = italy_ma_yearly[italy_ma_yearly.index.year >= 2010]

# Combined data (for plotting)
df_plot = pd.concat([italy_gdp_yearly, italy_ma_yearly], axis=1)

# =========================================================
# Plot function (B&W, journal style) — not called by default
# =========================================================
def plot_italy_gdp(df, save_path="FinalFigures/Italy_GDP_Annual.png"):
    if df is None or df.empty:
        raise ValueError("df_plot is empty.")

    fig, ax = plt.subplots(figsize=(10, 6))  # ~5:3 aspect ratio

    # Plot lines (solid vs dashed)
    if "Italy GDP (Yearly)" in df.columns:
        ax.plot(df.index, df["Italy GDP (Yearly)"], linestyle="-",  linewidth=1.5, label="Italy GDP (Yearly)")
    if "Italy GDP (MA-5)" in df.columns:
        ax.plot(df.index, df["Italy GDP (MA-5)"], linestyle="--", linewidth=1.5, label="Italy GDP (MA-5)")

    # Labels & ticks (12 pt)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("GDP (Bn€)",  fontsize=12)
    ax.set_xticks(df.index)
    ax.set_xticklabels(pd.to_datetime(df.index).year, fontsize=12)
    ax.tick_params(axis="y", labelsize=12)

    # Grid: horizontal only
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)

    # Minimal frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend: inside, thin border, white background (still monochrome)
    leg = ax.legend(loc="upper left", frameon=True, fontsize=10, facecolor="white", edgecolor="black")
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()

    # Save figure (creates folder if missing)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax

# --------------------------------
# Do NOT plot unless you set True
# --------------------------------
PLOT = True
if PLOT:
    plot_italy_gdp(df_plot)

#%%
#FIGURE 3
 

# -*- coding: utf-8 -*-
"""
Academic B&W style bar chart for country weights.

Update requested:
- No outlines (no edges) on bars for BOTH capped and uncapped.
  -> Bars use facecolor only; edgecolor='none', linewidth=0.0.
  -> Legend patches also without outlines for visual consistency.

Monochrome only; Computer Modern/CMU Serif; subtle dotted horizontal grid.
Does NOT plot by default (PLOT = False). Set True explicitly to render.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# -------------------------------
# Minimal global style (B&W only)
# -------------------------------
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

    # Grid (horizontal only; subtle dotted)
    "grid.color": "0.85",
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
})

# Neutral grays (monochrome only)
BAR_GRAY   = "#7f7f7f"  # use for CAPPED
LIGHT_GRAY = "#bfbfbf"  # use for UNCAPPED

# --- Params ---
cap_limit   = 0.25
target_date = "2024-12-31"  # any day in the month is fine
target_period = pd.Period(pd.to_datetime(target_date), freq="M")

# --- Helper: forgiving CSV loader to handle stray spaces in paths ---
def _read_csv_forgiving(path, **kw):
    try:
        return pd.read_csv(path, **kw)
    except FileNotFoundError:
        return pd.read_csv(path.replace(" ", ""), **kw)

# --- Load weights (wide) ---
pd_weight = _read_csv_forgiving("data /PDs+Weights.csv", index_col=0)
pd_weight.index = pd.to_datetime(pd_weight.index, errors="coerce").to_period("M")
row_w = pd_weight.loc[target_period]

w_cols = [c for c in pd_weight.columns if str(c).endswith("_Weights")]
weights = row_w[w_cols].astype(float).copy()
countries = [c.replace("_Weights", "").strip() for c in weights.index]
weights.index = countries
weights_sorted = weights.sort_values(ascending=False)

# --- Load capped flags (wide, 0/1 booleans; first col is 'YYYY_MM') ---
caps = _read_csv_forgiving("data /capped_flags.csv")
date_col = caps.columns[0]

# parse 'YYYY_MM' -> datetime (YYYY-MM-01) -> PeriodIndex('M')
caps[date_col] = caps[date_col].astype(str).str.replace("_", "-", regex=False)   # e.g., '2024-12'
caps["__dt__"] = pd.to_datetime(caps[date_col] + "-01", errors="coerce")
caps = caps.set_index("__dt__").drop(columns=[date_col])
caps.index = caps.index.to_period("M")

# tidy column labels to match weights
caps.columns = caps.columns.astype(str).str.strip()

# pick the row for the month; align to the countries we plot
row_c = caps.loc[target_period]
row_c = row_c.reindex(countries).fillna(0)
capped_flags = row_c.astype(float).gt(0.5)  # 1→True, 0→False

# =========================================================
# Plot function (B&W, journal style) — not called by default
# =========================================================
def plot_country_weights_bnw(weights_sorted, capped_flags, period, save_path="FinalFigures/Country_Weights_capped.png"):
    if weights_sorted is None or weights_sorted.empty:
        raise ValueError("weights_sorted is empty.")
    if capped_flags is None or capped_flags.empty:
        raise ValueError("capped_flags is empty.")

    fig, ax = plt.subplots(figsize=(10, 6))  # ~5:3 aspect ratio

    # Bars (NO OUTLINES):
    # - CAPPED   : BAR_GRAY (mid gray) fill
    # - UNCAPPED : LIGHT_GRAY (lighter gray) fill
    x = range(len(weights_sorted))
    idx = list(weights_sorted.index)

    for i, cty in enumerate(idx):
        is_capped = bool(capped_flags.get(cty, False))
        face = BAR_GRAY if is_capped else LIGHT_GRAY
        ax.bar(
            i, float(weights_sorted.loc[cty]),
            facecolor=face, edgecolor='none', linewidth=0.0
        )

    # Labels & ticks (12 pt)
    ax.set_xlabel("Country", fontsize=12)
    ax.set_ylabel("Weight",  fontsize=12)
    ax.set_xticks(list(x))
    ax.set_xticklabels(idx, fontsize=12, rotation=45, ha="right")
    ax.tick_params(axis="y", labelsize=12)

    # Horizontal grid only
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)

    # Minimal frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend: thin border on the legend box, patches WITHOUT outlines
    legend_handles = [
        Patch(facecolor=BAR_GRAY,   edgecolor='none', label="Capped"),
        Patch(facecolor=LIGHT_GRAY, edgecolor='none', label="Uncapped"),
    ]
    leg = ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=10,
                    facecolor="white", edgecolor="black")
    leg.get_frame().set_linewidth(0.8)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y*100):d}%"))

    fig.tight_layout()

    # Save figure (creates folder if missing)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax

# --------------------------------
# Do NOT plot unless you set True
# --------------------------------
PLOT = True  # keep False to avoid plotting by default
if PLOT:
    # Build a date-stamped filename (e.g., ".../Country_Weights_2024-12.png")
    save_name = f"FinalFigures/Country_Weights_{target_period.strftime('%Y-%m')}.png"
    plot_country_weights_bnw(weights_sorted, capped_flags, target_period, save_path=save_name)
    print(f"Saved: {save_name}")


# %% FIG 4 -------------
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

# Use this neutral mid-gray for "access"
BAR_GRAY = "#7f7f7f"

# -------------------------------
# Global minimalist B&W style
# -------------------------------
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
    "grid.color": "0.85",
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
})

# =========================================================
# Prepare data (expects DataFrame `MAdf` provided by caller)
#   rows = time (YYYY-MM or similar)
#   cols = countries
#   values = 0/1
# =========================================================
def _prep_ma_df(MAdf):
    df = MAdf.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        date_col = MAdf.columns[0]
        if not pd.to_datetime(MAdf[date_col], errors="coerce").notna().all():
            for cand in MAdf.columns:
                if 'date' in cand.lower() or 'time' in cand.lower():
                    date_col = cand
                    break
        df = df.set_index(pd.to_datetime(df[date_col], errors="coerce"))
        if date_col in df.columns:
            df = df.drop(columns=[date_col])

    df = df[~df.index.isna()].copy()
    df.index = df.index.to_period('M').to_timestamp('M')
    df = df.sort_index()
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0).clip(0, 1).astype(int)

    # Keep input column order but clean labels
    df.columns = [str(c).strip() for c in df.columns]
    return df

# =========================================================
# Enhanced plot: sorted + TOP marginal ONLY (no right panel)
# =========================================================
def plot_market_access_grid_enhanced_top_only(
    df,
    sort_by="intensity",   # "intensity" to sort countries by access share, or None to keep original order
    save_path="FinalFigures/Market_Access_Grid_Enhanced_TopOnly.png"
):
    """
    Monochrome, journal-style binary heatmap with:
      - Countries optionally sorted by access intensity (descending)
      - Top marginal line: number of countries with access per month
    No right-side per-country share panel.
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty.")

    # Sort countries by access share if requested
    if sort_by == "intensity":
        intensity = df.mean(axis=0)
        order = intensity.sort_values(ascending=False).index
        df = df[order]

    countries = df.columns.tolist()
    time_index = df.index
    n_c, n_t = len(countries), len(time_index)

    # Layout: top marginal small, main heatmap large
    fig_h = max(6, 0.42 * n_c)
    fig_w = 12
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[3, 22], hspace=0.08)

    ax_top  = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0])

    # ---- Main heatmap: 0 = white (no access), 1 = grey (access) ----
    cmap = ListedColormap(["white", BAR_GRAY])
    ax_main.imshow(df.values.T, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)

    # Y: countries
    ax_main.set_yticks(np.arange(n_c))
    ax_main.set_yticklabels(countries, fontsize=12)

    # X: yearly ticks
    ax_main.set_xlim(-0.5, n_t - 0.5)
    start_year, end_year = time_index.min().year, time_index.max().year
    year_starts = pd.date_range(f"{start_year}-01-01", f"{end_year}-01-01", freq="YS")
    ti_vals = time_index.values
    xpos = np.searchsorted(ti_vals, year_starts.values, side="left")
    xpos = np.clip(xpos, 0, n_t - 1)
    unique_mask = np.concatenate(([True], np.diff(xpos) != 0))
    xpos = xpos[unique_mask]
    year_labels = year_starts.year.astype(str).to_numpy()[unique_mask]

    # Thin labels if too many
    max_labels = 12
    if len(xpos) > max_labels:
        step = int(np.ceil(len(xpos) / max_labels))
        xpos = xpos[::step]
        year_labels = year_labels[::step]

    ax_main.set_xticks(xpos)
    ax_main.set_xticklabels(year_labels, fontsize=12)
    ax_main.set_xlabel("Year", fontsize=12)

    # Subtle separators
    for x in xpos:
        ax_main.axvline(x - 0.5, color="0.85", linestyle=":", linewidth=0.6)
    for y in np.arange(n_c + 1) - 0.5:
        ax_main.axhline(y, color="0.85", linestyle=":", linewidth=0.6)

    # Minimal frame
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)

    # Legend (access = grey)
    leg = ax_main.legend(
        handles=[
            Patch(facecolor=BAR_GRAY, edgecolor="black", label="Access (1)"),
            Patch(facecolor="white",   edgecolor="black", label="No access (0)"),
        ],
        loc="upper left", frameon=True, fontsize=10, facecolor="white", edgecolor="black"
    )
    leg.get_frame().set_linewidth(0.8)

    # ---- Top marginal: # countries with access each month ----
    monthly_total = df.sum(axis=1)
    ax_top.plot(np.arange(n_t), monthly_total.values, linestyle="-", linewidth=1.5, color="black")
    ax_top.set_xlim(-0.5, n_t - 0.5)
    ax_top.set_ylabel("# Countries", fontsize=12)

    # Share yearly ticks with main; hide top x labels
    ax_top.set_xticks(xpos)
    ax_top.set_xticklabels([])
    for x in xpos:
        ax_top.axvline(x - 0.5, color="0.85", linestyle=":", linewidth=0.6)

    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.grid(axis="y", which="major")

    fig.tight_layout()

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig

# --------------------------------
# Do NOT plot unless you set True
# --------------------------------
PLOT = True
if PLOT:
    df_ready = _prep_ma_df(MAdf)
    plot_market_access_grid_enhanced_top_only(
        df_ready,
        sort_by="intensity",  # or None to keep original country order
        save_path="FinalFigures/Market_Access_Grid.png"
    )
    print("Saved: FinalFigures/Market_Access_Grid.png")


# %% FIG 5 ----------------
import os
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Global minimalist B&W style
# -------------------------------
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
    "grid.color": "0.85",
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
})

# ===========================
# Data (as provided)
# ===========================
x_vals = np.array([
    0.01, 0.02, 0.02, 0.04, 0.04, 0.05, 0.1, 0.1, 0.1, 0.12,
    0.3, 0.3, 0.4, 0.4, 0.4, 0.5, 2.5, 10, 45
], dtype=float)

y_vals = np.array([
    0.5, 1, 1, 2, 2, 3, 4, 5, 6, 6,
    10, 10, 11, 15, 15, 18, 30, 40, 75
], dtype=float)

# ===========================
# Model curve: y = 7.9265 ln(x) + 25.791
# ===========================
def model_fn(x):
    return 7.9265 * np.log(x) + 25.791

x_min = max(np.min(x_vals[x_vals > 0]), 1e-6)
x_max = np.max(x_vals)
x_line = np.logspace(np.log10(x_min), np.log10(x_max), 400)
y_line = model_fn(x_line)

# ===========================
# Plot function (not auto-run)
# ===========================
def plot_esbies_pd_scatter(
    x, y, x_line, y_line,
    save_path="FinalFigures/ESBIES_PD_Scatter.png"
):
    fig, ax = plt.subplots(figsize=(8.5, 5.5))  # ~5:3

    # Scatter: SOLID BLACK filled circles
    ax.scatter(
        x, y,
        s=28, facecolors="black", edgecolors="black", linewidths=0.8,
        label="Data"
    )

    # Model curve: black dashed line with equation + R^2 in legend
    ax.plot(
        x_line, y_line,
        linestyle="--", linewidth=1.5, color="black",
        label=r"$y = 7.9265\,\ln(x) + 25.791$" + "\n" + r"$R^2 = 0.8666$"
    )

    # Labels
    ax.set_xlabel("PDs under Normal State (%)", fontsize=12)
    ax.set_ylabel("PDs under Mild State (%)", fontsize=12)
    ax.tick_params(axis="both", labelsize=12)

    # Grid: horizontal only
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)

    # Minimal frame
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    leg = ax.legend(loc="upper left", frameon=True, fontsize=10,
                    facecolor="white", edgecolor="black")
    leg.get_frame().set_linewidth(0.8)
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y):d}%"))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{int(y):d}%"))
    
    fig.tight_layout()

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax

# -------------------------------
# Do NOT plot unless you set True
# -------------------------------
PLOT = True
if PLOT:
    plot_esbies_pd_scatter(x_vals, y_vals, x_line, y_line)
    print("Saved: FinalFigures/ESBIES_PD_Scatter.png")





# %%
