from pathlib import Path

import pandas as pd


import sys
sys.path.append(str(Path(__file__).parent))

from theory_benchmark_utils import (
    aggregate_subject_benchmark_to_group,
    compute_subject_theory_benchmark,
)

# --- NEW: allocation/Hahn framing ---
RUN_NEW_ANALYSES = True   # set False to skip the additional analyses written to analysis/output/new/
# --- END NEW ---


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR.parent / "input"
OUTPUT_DIR = BASE_DIR.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
INFILE = INPUT_DIR / "group_lv_panel.csv"
SUBJ_INFILE = INPUT_DIR / "subj_lv_panel.csv"
if not INFILE.exists():
    raise FileNotFoundError("group_lv_panel.csv not found in input directory.")
if not SUBJ_INFILE.exists():
    raise FileNotFoundError("subj_lv_panel.csv not found in input directory.")
print(f"[LOAD] 讀取：{INFILE}")

SOCIAL_COST_PER_EMISSION = 6.0

POLICY_OUTCOMES = [
    "emission_policy_diff",
    "production_policy_diff",
    "pi_policy_diff",
    "surplus_policy_diff",
]

SOCIAL_OUTCOMES = [
    "emission_diff",
    "surplus_diff",
    "production_diff",
]

DISTRIBUTION_OUTCOMES = [
    "within_group_sd_profit",
    "within_group_sd_pi_policy_diff",
    "dominant_profit_share",
]


def add_policy_rate_vars(df):
    df["policy_rate_low"] = (
        ((df["is_tax"] == 1) & (df["tax_rate_low"] == 1))
        | ((df["is_trade"] == 1) & (df["eq_price_low"] == 1))
    ).astype(int)
    df["policy_rate_high"] = (
        ((df["is_tax"] == 1) & (df["tax_rate_high"] == 1))
        | ((df["is_trade"] == 1) & (df["eq_price_high"] == 1))
    ).astype(int)
    df["policy_rate"] = 0.0
    policy_mask = (df["is_tax"] == 1) | (df["is_trade"] == 1)
    df.loc[policy_mask, "policy_rate"] = 6.0
    df.loc[df["policy_rate_low"] == 1, "policy_rate"] = 3.0
    df.loc[df["policy_rate_high"] == 1, "policy_rate"] = 9.0
    return df


df = pd.read_csv(INFILE, encoding="utf-8-sig")
subj_df = pd.read_csv(SUBJ_INFILE, encoding="utf-8-sig")
if "app" in subj_df.columns:
    subj_df = subj_df.query("app != 'Stage_MUDA'").copy()
subj_df = compute_subject_theory_benchmark(subj_df)
for col in [
    "is_tax",
    "is_trade",
    "tax_rate_low",
    "tax_rate_high",
    "eq_price_low",
    "eq_price_high",
    "is_dominant",
]:
    if col not in subj_df.columns:
        subj_df[col] = 0
subj_df = add_policy_rate_vars(subj_df)
subj_df["actual_production_profit"] = (
    pd.to_numeric(subj_df["revenue"], errors="coerce")
    - pd.to_numeric(subj_df["total_cost"], errors="coerce")
)
subj_df["actual_clean_surplus"] = (
    subj_df["actual_production_profit"]
    - pd.to_numeric(subj_df["emission"], errors="coerce") * 6
)
subj_df["realized_profit"] = pd.to_numeric(subj_df["net_profit"], errors="coerce")
subj_df["policy_private_profit"] = (
    subj_df["actual_production_profit"]
    - pd.to_numeric(subj_df["policy_rate"], errors="coerce").fillna(0)
    * pd.to_numeric(subj_df["emission"], errors="coerce")
)
subj_df["pi_policy_diff"] = (
    subj_df["policy_private_profit"]
    - pd.to_numeric(subj_df["pi_policy"], errors="coerce")
)
subj_df["dominant_realized_profit"] = subj_df["realized_profit"].where(
    subj_df["is_dominant"] == 1,
    0.0,
)
group_benchmark = aggregate_subject_benchmark_to_group(subj_df)
df = df.merge(group_benchmark, on=["session_code", "app", "round"], how="left")
group_actual = (
    subj_df.groupby(["session_code", "app", "round"], dropna=False)
    .agg(
        actual_production_profit=("actual_production_profit", "sum"),
        actual_clean_surplus=("actual_clean_surplus", "sum"),
        within_group_sd_profit=("realized_profit", "std"),
        within_group_sd_pi_policy_diff=("pi_policy_diff", "std"),
        group_realized_profit=("realized_profit", "sum"),
        dominant_realized_profit=("dominant_realized_profit", "sum"),
    )
    .reset_index()
)
group_actual["dominant_profit_share"] = (
    group_actual["dominant_realized_profit"]
    / group_actual["group_realized_profit"].where(group_actual["group_realized_profit"] != 0)
)
df = df.merge(group_actual, on=["session_code", "app", "round"], how="left")
df["raw_surplus"] = df["surplus"]
df["surplus"] = df["actual_clean_surplus"]
df = add_policy_rate_vars(df)

required_policy_cols = {"E_policy", "Q_policy", "Pi_policy"}
missing_policy_cols = sorted(required_policy_cols - set(df.columns))
if missing_policy_cols:
    raise ValueError(
        "group_lv_panel.csv is missing policy benchmark columns. "
        "Rerun build/code/gen_subj_and_group_panel.py first: "
        + ", ".join(missing_policy_cols)
    )

# 新增 *_diff 欄位
df["emission_diff"] = df["emission"] - df["E_soc"]
df["surplus_diff"] = df["surplus"] - df["surplus_soc"]
df["production_diff"] = df["production"] - df["Q_soc"]
df["benchmark_emission_diff"] = df["benchmark_emission"] - df["E_soc"]
df["benchmark_surplus_diff"] = df["benchmark_surplus"] - df["surplus_soc"]
df["benchmark_production_diff"] = df["benchmark_production"] - df["Q_soc"]
df["emission_policy_diff"] = df["emission"] - df["E_policy"]
df["production_policy_diff"] = df["production"] - df["Q_policy"]
policy_rate = pd.to_numeric(df["policy_rate"], errors="coerce").fillna(0)
emission = pd.to_numeric(df["emission"], errors="coerce")
E_policy = pd.to_numeric(df["E_policy"], errors="coerce")
Pi_policy = pd.to_numeric(df["Pi_policy"], errors="coerce")
df["policy_private_profit"] = df["actual_production_profit"] - policy_rate * emission
df["pi_policy_diff"] = df["policy_private_profit"] - Pi_policy
df["surplus_policy"] = (
    Pi_policy + policy_rate * E_policy - SOCIAL_COST_PER_EMISSION * E_policy
)
df["surplus_policy_diff"] = df["surplus"] - df["surplus_policy"]

BENCHMARK_Y_MAP = {
    "emission_diff": "benchmark_emission_diff",
    "surplus_diff": "benchmark_surplus_diff",
    "production_diff": "benchmark_production_diff",
}


    # MODELS: (formula, sample) tuple

MODELS = [
    # emission_diff
    ("emission_diff ~ is_tax + is_trade", "all"),
    ("emission_diff ~ tax_rate_low + tax_rate_high", "is_tax"),
    ("emission_diff ~ is_equal + eq_price_low + eq_price_high", "is_trade"),
    ("emission_diff ~ is_equal + eq_price_low + eq_price_high + is_equal:eq_price_low + is_equal:eq_price_high", "is_trade"),
    # surplus_diff
    ("surplus_diff ~ is_tax + is_trade", "all"),
    ("surplus_diff ~ tax_rate_low + tax_rate_high", "is_tax"),
    ("surplus_diff ~ is_equal + eq_price_low + eq_price_high", "is_trade"),
    ("surplus_diff ~ is_equal + eq_price_low + eq_price_high + is_equal:eq_price_low + is_equal:eq_price_high", "is_trade"),
    # production_diff
    ("production_diff ~ is_tax + is_trade", "all"),
    ("production_diff ~ tax_rate_low + tax_rate_high", "is_tax"),
    ("production_diff ~ is_equal + eq_price_low + eq_price_high", "is_trade"),
    ("production_diff ~ is_equal + eq_price_low + eq_price_high + is_equal:eq_price_low + is_equal:eq_price_high", "is_trade"),
]

for outcome in SOCIAL_OUTCOMES:
    MODELS.extend([
        (f"{outcome} ~ tax_rate_low + tax_rate_high", "all"),
        (f"{outcome} ~ is_equal + eq_price_low + eq_price_high", "all"),
        (
            f"{outcome} ~ is_equal + eq_price_low + eq_price_high + "
            "is_equal:eq_price_low + is_equal:eq_price_high",
            "all",
        ),
        (f"{outcome} ~ tax_rate_low + tax_rate_high + round", "is_tax"),
    ])

for outcome in POLICY_OUTCOMES:
    MODELS.extend([
        (f"{outcome} ~ tax_rate_low + tax_rate_high", "is_tax"),
        (f"{outcome} ~ tax_rate_low + tax_rate_high + round", "is_tax"),
        (f"{outcome} ~ is_equal + eq_price_low + eq_price_high", "is_trade"),
        (
            f"{outcome} ~ is_equal + eq_price_low + eq_price_high + "
            "is_equal:eq_price_low + is_equal:eq_price_high",
            "is_trade",
        ),
        (f"{outcome} ~ tax_rate_low + tax_rate_high", "policy"),
        (f"{outcome} ~ is_equal + eq_price_low + eq_price_high", "policy"),
        (
            f"{outcome} ~ is_equal + eq_price_low + eq_price_high + "
            "is_equal:eq_price_low + is_equal:eq_price_high",
            "policy",
        ),
        (
            f"{outcome} ~ is_trade + policy_rate_low + policy_rate_high + "
            "is_trade:policy_rate_low + is_trade:policy_rate_high",
            "policy",
        ),
    ])

for outcome in DISTRIBUTION_OUTCOMES:
    MODELS.extend([
        (f"{outcome} ~ is_equal + eq_price_low + eq_price_high", "is_trade"),
        (
            f"{outcome} ~ is_equal + eq_price_low + eq_price_high + "
            "is_equal:eq_price_low + is_equal:eq_price_high",
            "is_trade",
        ),
    ])

from collections import defaultdict
from reg_utils import reg_func

# 定義額外的檢定函數
def group_lv_tests(res, idx, label):
    """對 group level 迴歸進行額外的 Wald 檢定"""
    lines = []
    params = res.params.index.astype(str)
    
    # Tax rate 檢定: tax_rate_low = tax_rate_high
    if ('tax_rate_low' in params.values) and ('tax_rate_high' in params.values):
        wald = res.wald_test("tax_rate_low - tax_rate_high = 0")
        lines.append(f"Wald test (tax_rate_low = tax_rate_high): chi2={float(wald.statistic):.3f}, df={int(wald.df_denom)}, p={float(wald.pvalue):.4f}")
    
    # Eq price 檢定: eq_price_low = eq_price_high
    if ('eq_price_low' in params.values) and ('eq_price_high' in params.values):
        wald = res.wald_test("eq_price_low - eq_price_high = 0")
        lines.append(f"Wald test (eq_price_low = eq_price_high): chi2={float(wald.statistic):.3f}, df={int(wald.df_denom)}, p={float(wald.pvalue):.4f}")
    
    # 交互作用項的檢定
    if ('is_equal:eq_price_low' in params.values) and ('is_equal:eq_price_high' in params.values):
        # 檢定交互作用項是否聯合顯著
        wald = res.wald_test([
            "is_equal:eq_price_low = 0",
            "is_equal:eq_price_high = 0"
        ])
        lines.append(f"Wald test (interaction terms jointly zero): chi2={float(wald.statistic):.3f}, df={int(wald.df_denom)}, p={float(wald.pvalue):.4f}")

    if {
        "is_trade",
        "policy_rate_low",
        "policy_rate_high",
        "is_trade:policy_rate_low",
        "is_trade:policy_rate_high",
    }.issubset(set(params.values)):
        for expr, desc in [
            ("is_trade = 0", "Trade - Tax at policy rate 6"),
            ("is_trade + is_trade:policy_rate_low = 0", "Trade - Tax at policy rate 3"),
            ("is_trade + is_trade:policy_rate_high = 0", "Trade - Tax at policy rate 9"),
        ]:
            tt = res.t_test(expr)
            lines.append(
                f"{desc}: effect={tt.effect.item():.3f}, SE={tt.sd.item():.3f}, "
                f"t={tt.tvalue.item():.3f}, p={tt.pvalue.item():.4f}"
            )
        wald = res.wald_test([
            "is_trade:policy_rate_low = 0",
            "is_trade:policy_rate_high = 0",
        ])
        lines.append(
            "Wald test (trade-tax difference does not vary by policy rate): "
            f"chi2={float(wald.statistic):.3f}, df={int(wald.df_denom)}, p={float(wald.pvalue):.4f}"
        )
    
    return lines if lines else None

# 根據 sample 類型分組處理
for sample_type in ["all", "is_tax", "is_trade", "policy"]:
    print(f"\n{'='*80}")
    print(f"處理樣本類型: {sample_type}")
    print(f"{'='*80}")
    
    # 篩選該樣本類型的模型
    sample_models = [(formula, sample) for formula, sample in MODELS if sample == sample_type]
    
    if not sample_models:
        continue
    
    # 準備該樣本的資料
    if sample_type == "all":
        df_sample = df.copy()
        sample_label = "all"
    elif sample_type == "is_tax":
        df_sample = df[df['is_tax'] == 1].copy()
        sample_label = "tax"
    elif sample_type == "is_trade":
        df_sample = df[df['is_trade'] == 1].copy()
        sample_label = "trade"
    elif sample_type == "policy":
        df_sample = df[(df['is_tax'] == 1) | (df['is_trade'] == 1)].copy()
        sample_label = "policy"
    
    print(f"樣本大小: {len(df_sample)} 筆觀察值")
    
    # 建立 y_to_formulas 結構
    y_to_formulas = defaultdict(list)
    for formula, _ in sample_models:
        y = formula.split("~")[0].strip()
        y_to_formulas[y].append(formula)
    
    # 執行迴歸並輸出結果
    output_prefix = str(OUTPUT_DIR / f"group_lv_{sample_label}")
    reg_func(
        df_sample,
        y_to_formulas,
        output_prefix,
        cluster_col="session_code",
        extra_effects_func=group_lv_tests,
        benchmark_y_map=BENCHMARK_Y_MAP,
    )

print(f"\n{'='*80}")
print("所有迴歸分析完成！")
print(f"{'='*80}")


# ============================================================
# --- NEW: allocation/Hahn framing ---
import os
import numpy as np
# Efficiency index and loss decomposition per session-round, late-round versions of Tables 1 and 3,
# per-round trends, and session-mean plots. Outputs go to analysis/output/new/.
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reg_utils import new_output_dir, new_reg_table
from theory_benchmark_utils import _parse_disturbance_values

_EQ = "eq_price_low + eq_price_high"
_ALLOC_HYP = ["is_equal = 0", "is_equal + is_equal:eq_price_low = 0", "is_equal + is_equal:eq_price_high = 0",
              "is_equal:eq_price_low = 0, is_equal:eq_price_high = 0"]


def _firm_units(row):
    """Marginal profit of each output unit (price - realised marginal cost) and emissions per unit."""
    price = float(row["market_price"]); a = float(row["marginal_cost_coefficient"])
    e = int(round(float(row["carbon_emission_per_unit"]))); qmax = int(float(row["max_production"]))
    dist = _parse_disturbance_values(row["disturbance_values"])
    n = min(qmax, len(dist))
    mp = np.array([price - (a * (j + 1) + float(dist[j])) for j in range(n)], dtype=float)
    return mp, e


def _firm_best_given_permits(mp, e, permits):
    """Profit-maximising output when permits are free but capped at `permits` (game rule: produce while mp > 0)."""
    q = 0; prof = 0.0
    for j, m in enumerate(mp):
        if m <= 0 or e * (j + 1) > permits + 1e-9:
            break
        q += 1; prof += m
    return q, prof


def _planner(units, budget, social_cost=None):
    """Cost-effective allocation of an emissions budget across firms (greedy by marginal profit per permit; units are
    taken in order within a firm, which is optimal because marginal profit is decreasing in output).
    With `social_cost`, the budget is ignored and only units with marginal profit > social_cost x e are taken."""
    ptr = [0] * len(units); used = 0.0; prof = 0.0
    while True:
        best = None
        for i, (mp, e) in enumerate(units):
            j = ptr[i]
            if j >= len(mp):
                continue
            m = mp[j]
            if social_cost is not None:
                ok = m > social_cost * e
            else:
                ok = (m > 0) and (used + e <= budget + 1e-9)
            if ok and (best is None or m / e > best[0]):
                best = (m / e, i)
        if best is None:
            break
        i = best[1]; mp, e = units[i]
        prof += mp[ptr[i]]; used += e; ptr[i] += 1
    return prof, used


def efficiency_decomposition(subj_df, social_cost=6.0):
    """Session-round table of realised vs benchmark production profit and surplus, with the loss decomposition."""
    rows = []
    keys = ["session_code", "app", "round"]
    for (s, app, r), x in subj_df.groupby(keys):
        if app not in ("Stage_CarbonTax", "Stage_CarbonTrading"):
            continue
        units = [_firm_units(row) for _, row in x.iterrows()]
        prod_act = float((pd.to_numeric(x["revenue"]) - pd.to_numeric(x["total_cost"])).sum())
        E_act = float(pd.to_numeric(x["emission"]).sum())
        prof_soc, E_soc = _planner(units, None, social_cost=social_cost)
        rec = dict(session_code=s, app=app, round=r, is_trade=int(app == "Stage_CarbonTrading"), is_tax=int(app == "Stage_CarbonTax"),
                   is_equal=int(x["is_equal"].iloc[0]), eq_price_low=int(x["eq_price_low"].iloc[0]) if app == "Stage_CarbonTrading" else int(x["tax_rate_low"].iloc[0]),
                   eq_price_high=int(x["eq_price_high"].iloc[0]) if app == "Stage_CarbonTrading" else int(x["tax_rate_high"].iloc[0]),
                   prod_profit_actual=prod_act, emissions_actual=E_act, surplus_actual=prod_act - social_cost * E_act,
                   prod_profit_soc=prof_soc, emissions_soc=E_soc, surplus_soc=prof_soc - social_cost * E_soc)
        prof_optE, _ = _planner(units, E_act)                                   # best allocation of the realised aggregate emissions
        rec["prod_profit_opt_at_actual_E"] = prof_optE
        rec["cross_firm_loss"] = prof_optE - prod_act                            # misallocation across firms at given total emissions (ECU)
        rec["aggregate_wedge_surplus"] = rec["surplus_soc"] - (prof_optE - social_cost * E_act)  # loss from the wrong aggregate emissions
        if app == "Stage_CarbonTrading":
            init = pd.to_numeric(x["initial_permits"]).to_numpy(); fin = pd.to_numeric(x["final_permits"]).to_numpy()
            cap = float(init.sum())
            prof_cap, E_cap = _planner(units, cap)
            prof_aut = sum(_firm_best_given_permits(mp, e, p)[1] for (mp, e), p in zip(units, init))
            E_aut = sum(e * _firm_best_given_permits(mp, e, p)[0] for (mp, e), p in zip(units, init))
            hold = [_firm_best_given_permits(mp, e, p) for (mp, e), p in zip(units, fin)]
            prof_hold = sum(h[1] for h in hold); E_hold = sum(e * h[0] for (mp, e), h in zip(units, hold))
            prof_optEh, _ = _planner(units, E_hold)
            rec.update(cap=cap, prod_profit_cap=prof_cap, emissions_cap_used=E_cap, surplus_cap=prof_cap - social_cost * E_cap,
                       prod_profit_autarky=prof_aut, emissions_autarky=E_aut, surplus_autarky=prof_aut - social_cost * E_aut,
                       prod_profit_holdings=prof_hold, emissions_holdings=E_hold, surplus_holdings=prof_hold - social_cost * E_hold,
                       prod_profit_opt_at_holdings_E=prof_optEh,
                       loss_total=prof_cap - prod_act, loss_unused_permits=prof_cap - prof_optEh,
                       loss_misallocation=prof_optEh - prof_hold, loss_production_error=prof_hold - prod_act,
                       unused_permits=cap - E_act)
            denom = prof_cap - prof_aut
            rec["potential_gains_from_trade"] = denom
            rec["potential_gains_share"] = denom / prof_cap if prof_cap > 0 else np.nan
            # the gains-from-trade index is only meaningful when trade can add something: require potential gains >= 5% of the cap optimum
            rec["efficiency_index"] = (prod_act - prof_aut) / denom if (prof_cap > 0 and denom >= 0.05 * prof_cap) else np.nan
            rec["profit_ratio_to_cap_optimum"] = prod_act / prof_cap if prof_cap > 0 else np.nan
            rec["surplus_ratio_to_soc"] = rec["surplus_actual"] / rec["surplus_soc"] if rec["surplus_soc"] > 0 else np.nan
            for k in ["loss_unused_permits", "loss_misallocation", "loss_production_error"]:
                rec[k + "_share"] = rec[k] / prof_cap if prof_cap > 0 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def run_new_group_analyses(df_group, subj_df, output_dir, cluster_col="session_code"):
    new_dir = new_output_dir(output_dir)
    tidy_all = []
    # ---- A8: efficiency index and decomposition ----
    eff = efficiency_decomposition(subj_df)
    eff.to_csv(os.path.join(new_dir, "efficiency_decomposition.csv"), index=False, encoding="utf-8-sig")
    tr = eff[eff["is_trade"] == 1].copy()
    means = tr.groupby(["is_equal", "eq_price_low", "eq_price_high"]).agg(
        rounds=("round", "size"), efficiency_index=("efficiency_index", "mean"), rounds_with_index=("efficiency_index", "count"),
        potential_gains_share=("potential_gains_share", "mean"), profit_ratio_to_cap_optimum=("profit_ratio_to_cap_optimum", "mean"),
        loss_total=("loss_total", "mean"), loss_unused_permits=("loss_unused_permits", "mean"), loss_misallocation=("loss_misallocation", "mean"),
        loss_production_error=("loss_production_error", "mean"), unused_permits=("unused_permits", "mean"), cap=("cap", "mean"),
        surplus_actual=("surplus_actual", "mean"), surplus_cap=("surplus_cap", "mean"), surplus_soc=("surplus_soc", "mean")).reset_index()
    means.insert(0, "stage", "trade")
    mt = eff[eff["is_tax"] == 1].groupby(["eq_price_low", "eq_price_high"]).agg(
        rounds=("round", "size"), cross_firm_loss=("cross_firm_loss", "mean"), aggregate_wedge_surplus=("aggregate_wedge_surplus", "mean"),
        surplus_actual=("surplus_actual", "mean"), surplus_soc=("surplus_soc", "mean")).reset_index()
    mt.insert(0, "stage", "tax")
    pd.concat([means, mt], ignore_index=True).to_csv(os.path.join(new_dir, "efficiency_decomposition_means.csv"), index=False, encoding="utf-8-sig")
    f_eff = [f"{y} ~ is_equal*({_EQ}) + round" for y in ["profit_ratio_to_cap_optimum", "loss_unused_permits_share", "loss_misallocation_share", "loss_production_error_share", "unused_permits"]]
    tidy = new_reg_table(tr, f_eff, cluster_col, os.path.join(new_dir, "efficiency_reg.txt"),
                         "A8. Efficiency index and loss decomposition under cap-and-trade (session-round)",
                         ["Production profit = revenue - cost, summed over the 15 firms; benchmarks recomputed with each firm's realised cost draws.",
                          "prod_profit_cap = cost-effective allocation of the cap (greedy by marginal profit per permit); autarky = optimal production with initial permits only;",
                          "efficiency_index = (actual - autarky) / (cap optimum - autarky), the standard gains-from-trade-realised index, reported only when",
                          "potential gains from trade are at least 5% of the cap optimum (under grandfathering the initial allocation is often close to efficient);",
                          "profit_ratio_to_cap_optimum = actual production profit / cap optimum is defined for every round;",
                          "loss decomposition (shares of the cap optimum): unused permits (cap optimum - optimum at the emissions implied by final holdings),",
                          "misallocation (that optimum - optimal production given each firm's actual holdings), production error (holdings optimum - actual)."],
                         hypotheses={f: _ALLOC_HYP for f in f_eff})
    tidy.insert(0, "block", "A8_efficiency"); tidy_all.append(tidy)
    idx = tr.dropna(subset=["efficiency_index"])
    if len(idx) > 30:
        t_idx = new_reg_table(idx, [f"efficiency_index ~ is_equal*({_EQ}) + round"], cluster_col, os.path.join(new_dir, "efficiency_reg_index.txt"),
                              "A8. Gains-from-trade index (rounds with potential gains >= 5% of the cap optimum)", ["See efficiency_reg.txt for definitions."],
                              hypotheses={f"efficiency_index ~ is_equal*({_EQ}) + round": _ALLOC_HYP})
        t_idx.insert(0, "block", "A8_efficiency_index"); tidy_all.append(t_idx)
    pol = eff.copy(); pol["policy_rate_low"] = pol["eq_price_low"]; pol["policy_rate_high"] = pol["eq_price_high"]
    f_pol = ["cross_firm_loss ~ is_trade + policy_rate_low + policy_rate_high + is_trade:policy_rate_low + is_trade:policy_rate_high + round",
             "aggregate_wedge_surplus ~ is_trade + policy_rate_low + policy_rate_high + is_trade:policy_rate_low + is_trade:policy_rate_high + round"]
    tidy = new_reg_table(pol, f_pol, cluster_col, os.path.join(new_dir, "efficiency_reg_tax_vs_trade.txt"),
                         "A8b. Cross-firm misallocation loss and aggregate wedge: tax vs cap-and-trade at matched carbon prices (session-round)",
                         ["cross_firm_loss = optimal production profit at the realised aggregate emissions - actual production profit (ECU);",
                          "aggregate_wedge_surplus = social optimum surplus - surplus at the realised aggregate emissions allocated optimally.",
                          "Hypotheses: trade - tax at carbon price 6 / 3 / 9."],
                         hypotheses={f: ["is_trade = 0", "is_trade + is_trade:policy_rate_low = 0", "is_trade + is_trade:policy_rate_high = 0"] for f in f_pol})
    tidy.insert(0, "block", "A8_tax_vs_trade"); tidy_all.append(tidy)

    # ---- A11: Tables 1 and 3 specifications, full sample and rounds 7-12; trends by round ----
    g = df_group.copy()
    spec1 = ["emission_diff ~ is_tax + is_trade", "surplus_diff ~ is_tax + is_trade"]
    spec3 = [f"emission_diff ~ is_equal*({_EQ})", f"surplus_diff ~ is_equal*({_EQ})"]
    for label, gg in (("full_sample", g), ("rounds_7_12", g[g["round"] >= 7])):
        t1 = new_reg_table(gg, spec1, cluster_col, os.path.join(new_dir, f"robustness_group_table1_{label}.txt"),
                           f"A11/A12. Table 1 specification (all stages), {label}", ["emission_diff = emission - E_soc; surplus_diff = clean surplus - surplus_soc, as in the paper."],
                           hypotheses={f: ["is_tax - is_trade = 0"] for f in spec1})
        t1.insert(0, "block", f"A11_group_table1_{label}"); tidy_all.append(t1)
        t3 = new_reg_table(gg[gg["is_trade"] == 1], spec3, cluster_col, os.path.join(new_dir, f"robustness_group_table3_{label}.txt"),
                           f"A11/A12. Table 3 specification (trading rounds), {label}", ["Same specification as the paper's Table 3, with wild-cluster-bootstrap p-values."],
                           hypotheses={f: _ALLOC_HYP for f in spec3})
        t3.insert(0, "block", f"A11_group_table3_{label}"); tidy_all.append(t3)
    g["stage"] = np.select([g["is_tax"] == 1, g["is_trade"] == 1], ["tax", "trade"], default="control")
    g["arm"] = np.where(g["is_trade"] == 1, np.where(g["is_equal"] == 1, "trade-equal", "trade-grandfathering"), g["stage"])
    trends = g.groupby(["arm", "round"]).agg(n=("emission_diff", "size"), emission_diff=("emission_diff", "mean"), surplus_diff=("surplus_diff", "mean")).reset_index()
    trends.to_csv(os.path.join(new_dir, "trends_by_round.csv"), index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, y in zip(axes, ["emission_diff", "surplus_diff"]):
        for arm, sub in trends.groupby("arm"):
            ax.plot(sub["round"], sub[y], marker="o", label=arm)
        ax.axhline(0, color="grey", lw=0.8); ax.set_xlabel("round within stage"); ax.set_ylabel(y); ax.set_title(f"{y} by round")
    axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(new_dir, "trends_by_round.png"), dpi=200); plt.close(fig)

    # ---- A12: session means with 95% CIs (trading rounds) ----
    tg = g[g["is_trade"] == 1].copy(); tg["eq_price"] = np.where(tg["eq_price_low"] == 1, 3, np.where(tg["eq_price_high"] == 1, 9, 6))
    tg = tg.merge(tr[["session_code", "round", "efficiency_index"]], on=["session_code", "round"], how="left")
    sm = tg.groupby(["session_code", "is_equal", "eq_price"]).agg(emission_diff=("emission_diff", "mean"), surplus_diff=("surplus_diff", "mean"),
                                                                   efficiency_index=("efficiency_index", "mean")).reset_index()
    sm.to_csv(os.path.join(new_dir, "session_means.csv"), index=False, encoding="utf-8-sig")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, y in zip(axes, ["emission_diff", "surplus_diff", "efficiency_index"]):
        for k, (alloc, col) in enumerate(((0, "#227D51"), (1, "#51ABDD"))):
            for j, p in enumerate([3, 6, 9]):
                v = sm[(sm["is_equal"] == alloc) & (sm["eq_price"] == p)][y].dropna()
                xpos = j + (k - 0.5) * 0.3
                ax.scatter(np.full(len(v), xpos), v, color=col, alpha=0.6, s=18)
                if len(v) > 1:
                    m = v.mean(); ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
                    ax.errorbar(xpos, m, yerr=ci, fmt="D", color=col, capsize=4, markersize=7)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["eq_price 3", "eq_price 6", "eq_price 9"]); ax.set_title(f"{y}: session means (green = grandfathering, blue = equal)")
        ax.axhline(0, color="grey", lw=0.8)
    fig.tight_layout(); fig.savefig(os.path.join(new_dir, "session_means.png"), dpi=200); plt.close(fig)
    pd.concat(tidy_all, ignore_index=True).to_csv(os.path.join(new_dir, "wildboot_pvalues_group.csv"), index=False, encoding="utf-8-sig")
    print(f"[NEW] group-level analyses written to {new_dir}")
    return eff
# --- END NEW ---

# --- NEW: allocation/Hahn framing --- run the additional group-level analyses
if RUN_NEW_ANALYSES:
    run_new_group_analyses(df, subj_df, OUTPUT_DIR, cluster_col="session_code")
# --- END NEW ---
