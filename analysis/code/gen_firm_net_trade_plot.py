import os
import shutil

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from theory_benchmark_utils import compute_subject_theory_benchmark

# --- NEW: allocation/Hahn framing ---
RUN_NEW_ANALYSES = True   # set False to skip the additional analyses written to analysis/output/new/
# --- END NEW ---


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(BASE_DIR, "..", "input", "subj_lv_panel.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")
PROJECT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
DOC_INPUT_DIR = os.path.join(PROJECT_DIR, "doc", "input")

OUTPUT_PNG = "firm_net_trade_by_type_eq_price.png"
OUTPUT_CSV = "firm_net_trade_by_type_eq_price_summary.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)
# [Replication package] doc/ 不在 package 內時，略過複製到 doc/input


def infer_eq_price(df):
    eq_price = np.where(
        pd.to_numeric(df["eq_price_low"], errors="coerce").fillna(0).eq(1),
        3,
        np.where(
            pd.to_numeric(df["eq_price_high"], errors="coerce").fillna(0).eq(1),
            9,
            6,
        ),
    )
    return pd.Series(eq_price, index=df.index, dtype=int)


def copy_to_doc_input(src_path):
    if not os.path.isdir(DOC_INPUT_DIR):
        return
    try:
        shutil.copy(src_path, os.path.join(DOC_INPUT_DIR, os.path.basename(src_path)))
    except Exception as exc:
        print(f"Warning: Could not copy {os.path.basename(src_path)} to doc/input: {exc}")


if not os.path.exists(INPUT):
    raise FileNotFoundError("subj_lv_panel.csv not found in input directory.")

df = pd.read_csv(INPUT)
required_cols = {"initial_permits", "final_permits", "net_trade"}
missing_cols = sorted(required_cols - set(df.columns))
if missing_cols:
    raise ValueError(
        "subj_lv_panel.csv is missing required columns for net trade analysis: "
        + ", ".join(missing_cols)
    )

trade_df = df[(df["is_trade"] == 1) & (df["app"] == "Stage_CarbonTrading")].copy()
trade_df["initial_permits"] = pd.to_numeric(trade_df["initial_permits"], errors="coerce")
trade_df["final_permits"] = pd.to_numeric(trade_df["final_permits"], errors="coerce")
trade_df["net_trade"] = pd.to_numeric(trade_df["net_trade"], errors="coerce")
trade_df = trade_df.dropna(subset=["initial_permits", "final_permits", "net_trade"])
trade_df["eq_price"] = infer_eq_price(trade_df)

trade_df = compute_subject_theory_benchmark(trade_df)
trade_df["benchmark_net_trade"] = pd.to_numeric(trade_df["benchmark_net_trade"], errors="coerce")
trade_df["firm_type"] = np.where(trade_df["is_dominant"] == 1, "Dominant firms", "Small firms")
trade_df["allocation_label"] = np.where(trade_df["is_equal"] == 1, "Equal", "Grandfathering")
trade_df["x_group"] = "eq_price = " + trade_df["eq_price"].astype(str) + "\n" + trade_df["firm_type"]

firm_order = ["Dominant firms", "Small firms"]
price_order = [3, 6, 9]
allocation_order = [1, 0]
x_order = [f"eq_price = {price}\n{firm_type}" for price in price_order for firm_type in firm_order]

summary = (
    trade_df.groupby(["firm_type", "eq_price", "is_equal", "allocation_label"], dropna=False)
    .agg(
        actual_net_trade=("net_trade", "mean"),
        actual_net_trade_sd=("net_trade", "std"),
        theory_net_trade=("benchmark_net_trade", "mean"),
        n_obs=("net_trade", "size"),
    )
    .reset_index()
)
summary["actual_net_trade_se"] = summary["actual_net_trade_sd"] / np.sqrt(summary["n_obs"].clip(lower=1))
summary["actual_net_trade_ci95"] = 1.96 * summary["actual_net_trade_se"]
summary["x_group"] = "eq_price = " + summary["eq_price"].astype(int).astype(str) + "\n" + summary["firm_type"]
summary["x_group"] = pd.Categorical(summary["x_group"], categories=x_order, ordered=True)
summary = summary.sort_values(["x_group", "is_equal"]).reset_index(drop=True)

summary_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV)
summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
copy_to_doc_input(summary_path)
print(f"[OK] Summary saved to: {summary_path}")

bar_colors = {
    1: "#51ABDD",
    0: "#227D51",
}
theory_color = "#D62728"

fig, ax = plt.subplots(figsize=(14, 7.5))
x = np.arange(len(x_order))
width = 0.34
offsets = {
    1: -width / 2,
    0: width / 2,
}

for alloc in allocation_order:
    sub = summary[summary["is_equal"] == alloc].set_index("x_group").reindex(x_order)
    x_pos = x + offsets[alloc]
    ax.bar(
        x_pos,
        sub["actual_net_trade"],
        width=width * 0.92,
        color=bar_colors[alloc],
        yerr=sub["actual_net_trade_ci95"].fillna(0),
        error_kw=dict(ecolor="lightgray", lw=1, capsize=5, capthick=1),
        edgecolor="none",
        zorder=2,
    )
    for xi, yi in zip(x_pos, sub["theory_net_trade"]):
        if pd.isna(yi):
            continue
        ax.hlines(
            yi,
            xi - width * 0.28,
            xi + width * 0.28,
            colors=theory_color,
            linewidth=2.6,
            zorder=4,
        )

ax.axhline(0, color="#777777", linestyle="--", linewidth=1.2, zorder=1)
for sep in [1.5, 3.5]:
    ax.axvline(sep, color="#D0D0D0", linestyle=":", linewidth=1.0, zorder=1)

all_vals = pd.concat([summary["actual_net_trade"], summary["theory_net_trade"]]).dropna()
if not all_vals.empty:
    y_pad = max(0.5, (all_vals.max() - all_vals.min()) * 0.15)
    ax.set_ylim(all_vals.min() - y_pad, all_vals.max() + y_pad)

ax.set_xticks(x)
ax.set_xticklabels(x_order, fontsize=12)
ax.set_ylabel("Firm Net Trade", fontsize=16)
ax.set_title("Firm Net Trade in Trading Rounds", fontsize=18)
ax.tick_params(axis="y", labelsize=13)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

legend_handles = [
    mpatches.Patch(color=bar_colors[1], label="Equal"),
    mpatches.Patch(color=bar_colors[0], label="Grandfathering"),
    mlines.Line2D([], [], color=theory_color, marker="_", linestyle="None", markersize=18, markeredgewidth=2.6, label="Theory benchmark"),
]
ax.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.01),
    frameon=False,
    fontsize=12,
    ncol=3,
)

plt.tight_layout(rect=(0, 0, 1, 0.96))
plot_path = os.path.join(OUTPUT_DIR, OUTPUT_PNG)
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close(fig)
copy_to_doc_input(plot_path)
print(f"[OK] Plot saved to: {plot_path}")

# --- NEW: allocation/Hahn framing ---
# (i) the net-trade figure (actual vs competitive benchmark, with CIs) also saved under analysis/output/new/;
# (ii) unused permits (final_permits - emission) by firm type x allocation x eq_price.
if RUN_NEW_ANALYSES:
    from reg_utils import new_output_dir
    _new_dir = new_output_dir(OUTPUT_DIR)
    shutil.copy(plot_path, os.path.join(_new_dir, "fig_net_trade_actual_vs_theory.png"))
    trade_df["emission"] = pd.to_numeric(trade_df["emission"], errors="coerce")
    trade_df["unused"] = trade_df["final_permits"] - trade_df["emission"]
    _u = (trade_df.groupby(["firm_type", "eq_price", "is_equal"])["unused"].agg(["mean", "std", "size"]).reset_index())
    _u["ci95"] = 1.96 * _u["std"] / np.sqrt(_u["size"].clip(lower=1))
    _u["x_group"] = "eq_price = " + _u["eq_price"].astype(int).astype(str) + "\n" + _u["firm_type"]
    fig, ax = plt.subplots(figsize=(12, 6))
    for alloc in allocation_order:
        sub = _u[_u["is_equal"] == alloc].set_index("x_group").reindex(x_order)
        ax.bar(x + offsets[alloc], sub["mean"], width=width * 0.92, color=bar_colors[alloc], yerr=sub["ci95"].fillna(0),
               error_kw=dict(ecolor="lightgray", lw=1, capsize=5, capthick=1), edgecolor="none", zorder=2)
    ax.axhline(0, color="#777777", linestyle="--", linewidth=1.2, zorder=1)
    ax.set_xticks(x); ax.set_xticklabels(x_order, fontsize=11)
    ax.set_ylabel("Unused permits per firm (final holdings - emissions)", fontsize=13)
    ax.set_title("Permits left unused at the end of the round", fontsize=15)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(handles=[mpatches.Patch(color=bar_colors[1], label="Equal"), mpatches.Patch(color=bar_colors[0], label="Grandfathering")], frameon=False, fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(_new_dir, "fig_unused_permits.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[NEW] figures saved to {_new_dir}")
# --- END NEW ---
