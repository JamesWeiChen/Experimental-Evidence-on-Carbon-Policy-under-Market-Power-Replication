import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.append(os.path.dirname(__file__))
from reg_utils import reg_func
from theory_benchmark_utils import compute_subject_theory_benchmark

# --- NEW: allocation/Hahn framing ---
RUN_NEW_ANALYSES = True   # set False to skip the additional analyses written to analysis/output/new/
# --- END NEW ---


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "..", "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

INFILE = os.path.join(INPUT_DIR, "subj_lv_panel.csv")
if not os.path.exists(INFILE):
    raise FileNotFoundError("subj_lv_panel.csv not found in input directory.")
print(f"[LOAD] 讀取：{INFILE}")

SOCIAL_COST_PER_EMISSION = 6.0

POLICY_OUTCOMES = [
    "emission_policy_diff",
    "production_policy_diff",
    "pi_policy_diff",
    "surplus_policy_diff",
]


def _add_policy_rate_vars(df):
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


def policy_compare_tests(res, idx, label):
    """Report trade-vs-tax effects at the same policy carbon price."""
    params = res.params.index.astype(str)
    if (
        "is_trade" not in params.values
        or "policy_rate_low" not in params.values
        or "policy_rate_high" not in params.values
    ):
        return None

    lines = []
    tests = [("is_trade = 0", "Trade - Tax at policy rate 6")]
    if "is_trade:policy_rate_low" in params.values:
        tests.append(("is_trade + is_trade:policy_rate_low = 0", "Trade - Tax at policy rate 3"))
    if "is_trade:policy_rate_high" in params.values:
        tests.append(("is_trade + is_trade:policy_rate_high = 0", "Trade - Tax at policy rate 9"))

    for expr, desc in tests:
        tt = res.t_test(expr)
        lines.append(
            f"{desc}: effect={tt.effect.item():.3f}, SE={tt.sd.item():.3f}, "
            f"t={tt.tvalue.item():.3f}, p={tt.pvalue.item():.4f}"
        )

    if {
        "is_trade:policy_rate_low",
        "is_trade:policy_rate_high",
    }.issubset(set(params.values)):
        wald = res.wald_test([
            "is_trade:policy_rate_low = 0",
            "is_trade:policy_rate_high = 0",
        ])
        lines.append(
            "Wald test (trade-tax difference does not vary by policy rate): "
            f"chi2={float(wald.statistic):.3f}, p={float(wald.pvalue):.4f}"
        )

    return lines


def _load_subject_panel():
    df = pd.read_csv(INFILE, encoding="utf-8-sig")
    if "app" in df.columns:
        df = df.query("app != 'Stage_MUDA'").copy()

    df = compute_subject_theory_benchmark(df)

    df["emission_diff"] = df["emission"] - df["e_soc"]
    df["raw_net_profit_diff"] = df["net_profit"] - df["pi_soc"]
    # Use production-side profit for subject-level profit gaps. In trading rounds,
    # net_profit includes permit-trading cash flow and free permit endowment value.
    df["production_profit"] = (
        pd.to_numeric(df["revenue"], errors="coerce")
        - pd.to_numeric(df["total_cost"], errors="coerce")
    )
    df["production_profit_diff"] = df["production_profit"] - df["pi_soc"]
    df["production_diff"] = df["production"] - df["q_soc"]
    df["benchmark_emission_diff"] = df["benchmark_emission"] - df["e_soc"]
    df["benchmark_production_profit"] = (
        pd.to_numeric(df["market_price"], errors="coerce") * df["benchmark_production"]
        - df["benchmark_total_cost"]
    )
    df["benchmark_production_profit_diff"] = df["benchmark_production_profit"] - df["pi_soc"]
    df["benchmark_production_diff"] = df["benchmark_production"] - df["q_soc"]

    for col in [
        "is_tax",
        "is_trade",
        "is_dominant",
        "is_equal",
        "tax_rate",
        "tax_rate_low",
        "tax_rate_high",
        "carbon_multiplier",
        "eq_price_low",
        "eq_price_high",
    ]:
        if col not in df.columns:
            df[col] = 0

    required_policy_cols = {"e_policy", "q_policy", "pi_policy"}
    missing_policy_cols = sorted(required_policy_cols - set(df.columns))
    if missing_policy_cols:
        raise ValueError(
            "subj_lv_panel.csv is missing policy benchmark columns. "
            "Rerun build/code/gen_subj_and_group_panel.py first: "
            + ", ".join(missing_policy_cols)
        )

    df = _add_policy_rate_vars(df)
    df["emission_policy_diff"] = (
        pd.to_numeric(df["emission"], errors="coerce")
        - pd.to_numeric(df["e_policy"], errors="coerce")
    )
    df["production_policy_diff"] = (
        pd.to_numeric(df["production"], errors="coerce")
        - pd.to_numeric(df["q_policy"], errors="coerce")
    )
    emission = pd.to_numeric(df["emission"], errors="coerce")
    e_policy = pd.to_numeric(df["e_policy"], errors="coerce")
    pi_policy = pd.to_numeric(df["pi_policy"], errors="coerce")
    policy_rate = pd.to_numeric(df["policy_rate"], errors="coerce").fillna(0)
    # Clean private-payoff gap under the policy carbon price. This excludes fixed
    # permit endowment value and observed trading cash flow in cap-and-trade.
    df["policy_private_profit"] = df["production_profit"] - policy_rate * emission
    df["pi_policy_diff"] = df["policy_private_profit"] - pi_policy
    df["actual_clean_surplus"] = (
        df["production_profit"] - SOCIAL_COST_PER_EMISSION * emission
    )
    df["surplus_policy"] = (
        pi_policy + policy_rate * e_policy - SOCIAL_COST_PER_EMISSION * e_policy
    )
    df["surplus_policy_diff"] = df["actual_clean_surplus"] - df["surplus_policy"]

    return df


def _prepare_trade_only_panel(df):
    trade_mask = df["is_trade"] == 1
    if "app" in df.columns:
        trade_mask = trade_mask & (df["app"] == "Stage_CarbonTrading")
    trade_df = df.loc[trade_mask].copy()

    required_cols = {"net_trade", "initial_permits"}
    missing_cols = sorted(required_cols - set(trade_df.columns))
    if missing_cols:
        raise ValueError(
            "subj_lv_panel.csv is missing required columns for net_trade regressions: "
            + ", ".join(missing_cols)
        )

    trade_df["net_trade"] = pd.to_numeric(trade_df["net_trade"], errors="coerce")
    trade_df["initial_permits"] = pd.to_numeric(trade_df["initial_permits"], errors="coerce")
    trade_df["benchmark_net_trade"] = pd.to_numeric(
        trade_df["benchmark_net_trade"], errors="coerce"
    )
    trade_df = trade_df.dropna(
        subset=["net_trade", "initial_permits", "benchmark_net_trade"]
    ).copy()
    return trade_df


def _build_formula_map():
    y_to_formulas = defaultdict(list)

    subject_models = [
        "emission_diff ~ is_tax + is_trade + is_dominant",
        "emission_diff ~ is_tax + is_trade + is_dominant + is_dominant:is_tax + is_dominant:is_trade",
        "emission_diff ~ is_dominant + tax_rate_low + tax_rate_high",
        "emission_diff ~ is_dominant + tax_rate_low + tax_rate_high + is_dominant:tax_rate_low + is_dominant:tax_rate_high",
        "emission_diff ~ is_dominant + eq_price_low + eq_price_high",
        "emission_diff ~ is_dominant + is_dominant:eq_price_low + is_dominant:eq_price_high",
        "emission_diff ~ is_equal + is_dominant + eq_price_low + eq_price_high",
        "emission_diff ~ is_equal * is_dominant + eq_price_low + eq_price_high",
        "emission_diff ~ is_equal * is_dominant + initial_permits + eq_price_low + eq_price_high",
        "production_profit_diff ~ is_tax + is_trade + is_dominant",
        "production_profit_diff ~ is_tax + is_trade + is_dominant + is_dominant:is_tax + is_dominant:is_trade",
        "production_profit_diff ~ is_dominant + tax_rate_low + tax_rate_high",
        "production_profit_diff ~ is_dominant + tax_rate_low + tax_rate_high + is_dominant:tax_rate_low + is_dominant:tax_rate_high",
        "production_profit_diff ~ is_dominant + eq_price_low + eq_price_high",
        "production_profit_diff ~ is_dominant + is_dominant:eq_price_low + is_dominant:eq_price_high",
        "production_profit_diff ~ is_equal + is_dominant + eq_price_low + eq_price_high",
        "production_profit_diff ~ is_equal * is_dominant + eq_price_low + eq_price_high",
        "production_profit_diff ~ is_equal * is_dominant + initial_permits + eq_price_low + eq_price_high",
        "production_diff ~ is_tax + is_trade + is_dominant",
        "production_diff ~ is_tax + is_trade + is_dominant + is_dominant:is_tax + is_dominant:is_trade",
        "production_diff ~ is_dominant + tax_rate_low + tax_rate_high",
        "production_diff ~ is_dominant + tax_rate_low + tax_rate_high + is_dominant:tax_rate_low + is_dominant:tax_rate_high",
        "production_diff ~ is_dominant + eq_price_low + eq_price_high",
        "production_diff ~ is_dominant + is_dominant:eq_price_low + is_dominant:eq_price_high",
        "production_diff ~ is_equal + is_dominant + eq_price_low + eq_price_high",
        "production_diff ~ is_equal * is_dominant + eq_price_low + eq_price_high",
        "production_diff ~ is_equal * is_dominant + initial_permits + eq_price_low + eq_price_high",
    ]
    for formula in subject_models:
        y_to_formulas[formula.split("~")[0].strip()].append(formula)

    policy_rhs = [
        "is_dominant + tax_rate_low + tax_rate_high",
        "is_dominant + tax_rate_low + tax_rate_high + is_dominant:tax_rate_low + is_dominant:tax_rate_high",
        "is_equal + is_dominant + eq_price_low + eq_price_high",
        "is_equal + is_dominant + eq_price_low + eq_price_high + is_dominant:eq_price_low + is_dominant:eq_price_high",
        "is_equal * is_dominant + eq_price_low + eq_price_high",
        "is_equal * is_dominant + initial_permits + eq_price_low + eq_price_high",
        "is_trade + is_dominant + policy_rate_low + policy_rate_high + is_trade:policy_rate_low + is_trade:policy_rate_high",
        "is_trade * is_dominant + policy_rate_low + policy_rate_high + is_trade:policy_rate_low + is_trade:policy_rate_high",
    ]
    for outcome in POLICY_OUTCOMES:
        for rhs in policy_rhs:
            y_to_formulas[outcome].append(f"{outcome} ~ {rhs}")

    net_trade_rhs = [
        "is_equal + is_dominant + eq_price_low + eq_price_high",
        "is_equal * is_dominant + eq_price_low + eq_price_high",
        "is_equal + is_dominant + eq_price_low + eq_price_high + is_equal:eq_price_low + is_equal:eq_price_high",
        "is_equal + is_dominant + eq_price_low + eq_price_high + is_dominant:eq_price_low + is_dominant:eq_price_high",
        "is_equal * is_dominant * (eq_price_low + eq_price_high)",
        "is_equal * is_dominant + initial_permits + eq_price_low + eq_price_high",
        "is_equal * is_dominant * (eq_price_low + eq_price_high) + initial_permits",
    ]
    for outcome in ["net_trade", "net_trade_pct_gap", "net_trade_abs_pct_dev"]:
        for rhs in net_trade_rhs:
            y_to_formulas[outcome].append(f"{outcome} ~ {rhs}")

    return y_to_formulas


def _build_outcome_samples(df):
    trade_df = _prepare_trade_only_panel(df)
    policy_df = df.loc[(df["is_tax"] == 1) | (df["is_trade"] == 1)].copy()
    scaled_trade_df = trade_df.loc[trade_df["benchmark_net_trade"].abs() >= 1].copy()
    scaled_trade_df["benchmark_scale"] = scaled_trade_df["benchmark_net_trade"].abs()
    scaled_trade_df["net_trade_pct_gap"] = (
        (scaled_trade_df["net_trade"] - scaled_trade_df["benchmark_net_trade"])
        / scaled_trade_df["benchmark_scale"]
    )
    scaled_trade_df["net_trade_abs_pct_dev"] = (
        (scaled_trade_df["net_trade"] - scaled_trade_df["benchmark_net_trade"])
        / scaled_trade_df["benchmark_net_trade"]
    ).abs()

    print(f"[INFO] net_trade 樣本數: {len(trade_df)}")
    print(f"[INFO] 保留 |benchmark_net_trade| >= 1 後樣本數: {len(scaled_trade_df)}")

    outcome_samples = {
        "net_trade": trade_df,
        "net_trade_pct_gap": scaled_trade_df,
        "net_trade_abs_pct_dev": scaled_trade_df,
    }
    outcome_samples.update({outcome: policy_df for outcome in POLICY_OUTCOMES})
    return outcome_samples


df = _load_subject_panel()
all_y_to_formulas = _build_formula_map()
main_y_to_formulas = defaultdict(list)
policy_y_to_formulas = defaultdict(list)
for y, formulas in all_y_to_formulas.items():
    target = policy_y_to_formulas if y in POLICY_OUTCOMES else main_y_to_formulas
    target[y].extend(formulas)
outcome_samples = _build_outcome_samples(df)

benchmark_y_map = {
    "emission_diff": "benchmark_emission_diff",
    "production_profit_diff": "benchmark_production_profit_diff",
    "production_diff": "benchmark_production_diff",
    "net_trade": "benchmark_net_trade",
}

reg_func(
    df,
    main_y_to_formulas,
    os.path.join(OUTPUT_DIR, "subj_lv"),
    cluster_col="participant_code",
    benchmark_y_map=benchmark_y_map,
    y_to_df=outcome_samples,
)

reg_func(
    df,
    policy_y_to_formulas,
    os.path.join(OUTPUT_DIR, "subj_lv_policy"),
    cluster_col="participant_code",
    y_to_df=outcome_samples,
    extra_effects_func=policy_compare_tests,
)


# ============================================================
# --- NEW: allocation/Hahn framing ---
import numpy as np
# Theoretical positions (realised and deterministic), volume gaps, unused permits,
# profit decomposition, and late-round / session-clustered versions of Tables 5-6.
# Outputs go to analysis/output/new/.
# ============================================================
from reg_utils import new_output_dir, new_reg_table
from theory_benchmark_utils import theory_positions_det, hahn_benchmark_table

_EQ = "eq_price_low + eq_price_high"
_DOM_ALLOC_HYP = ["is_dominant = 0", "is_dominant + is_dominant:is_equal = 0", "is_dominant:is_equal = 0",
                  "is_equal = 0", "is_equal + is_dominant:is_equal = 0"]
_DOM_ALLOC_NOTES = ["Hypotheses: dominant - small under grandfathering; dominant - small under equal allocation; difference of the two;",
                    "equal - grandfathering for small firms; equal - grandfathering for dominant firms."]


def _eq_price_from_dummies(d):
    return np.where(d["eq_price_low"] == 1, 3, np.where(d["eq_price_high"] == 1, 9, 6))


def _prep_trade_rows(trade_df):
    td = trade_df.copy()
    td["eq_price"] = _eq_price_from_dummies(td)
    td["firm_type"] = np.where(td["is_dominant"] == 1, "dominant", "small")
    td["allocation"] = np.where(td["is_equal"] == 1, "equal", "grandfathering")
    td["eff_holdings"] = pd.to_numeric(td["benchmark_emission"], errors="coerce")        # competitive holdings at eq_price, realised draws
    td["theory_net"] = pd.to_numeric(td["benchmark_net_trade"], errors="coerce")
    for c in ["initial_permits", "final_permits", "net_trade", "emission", "net_profit", "production_profit", "benchmark_production_profit"]:
        td[c] = pd.to_numeric(td[c], errors="coerce")
    td["cap"] = td.groupby(["session_code", "round"])["initial_permits"].transform("sum")
    td["gap"] = td["net_trade"] - td["theory_net"]
    td["abs_gap"] = td["gap"].abs()
    big = td["theory_net"].abs() >= 1
    td["share_realised"] = np.where(big, td["net_trade"] / td["theory_net"].where(big), np.nan)
    td["wrong_direction"] = (big & (np.sign(td["net_trade"]) == -np.sign(td["theory_net"])) & (td["net_trade"] != 0)).astype(int)
    td["unused"] = td["final_permits"] - td["emission"]
    td["unused_pos"] = (td["unused"] > 0).astype(int)
    td["constrained"] = (td["final_permits"] < td["eff_holdings"]).astype(int)
    td["short_of_efficient"] = (td["eff_holdings"] - td["final_permits"]).clip(lower=0)
    td["trading_cf"] = td["net_profit"] - td["production_profit"]                       # = permits sold - permits bought (ECU); identity checked on raw data
    td["trading_cf_bench"] = -td["eq_price"] * td["theory_net"]                            # competitive: pay/receive eq_price per permit traded
    td["net_profit_bench"] = td["benchmark_production_profit"] + td["trading_cf_bench"]
    td["prod_gap"] = td["production_profit"] - td["benchmark_production_profit"]
    td["trade_gap"] = td["trading_cf"] - td["trading_cf_bench"]
    td["net_gap"] = td["net_profit"] - td["net_profit_bench"]
    return td


def _by_type_alloc_price(td, cols):
    return td.groupby(["firm_type", "allocation", "eq_price"]).agg(**cols).reset_index()


def run_new_subject_analyses(df, trade_df, output_dir, cluster_col="session_code"):
    new_dir = new_output_dir(output_dir)
    td = _prep_trade_rows(trade_df)
    tidy_all = []
    base_notes = ["Data: subj_lv_panel.csv, Carbon Trading rounds (150 firms x 12 rounds), competitive benchmarks at eq_price from",
                  "theory_benchmark_utils.compute_subject_theory_benchmark (realised cost draws). Standard errors clustered by session (10 clusters)."]

    # ---- A1: theoretical positions by cell (realised) merged with the deterministic table ----
    grp = ["group_dominant_mc", "market_price", "eq_price", "allocation", "firm_type"]
    pos = td.groupby(grp).agg(n=("initial_permits", "size"), initial_permits=("initial_permits", "mean"),
                              eff_holdings=("eff_holdings", "mean"), theory_net=("theory_net", "mean"),
                              actual_net_trade=("net_trade", "mean"), cap=("cap", "mean")).reset_index()
    pos = pos.rename(columns={"group_dominant_mc": "dominant_mc"})
    sr = td.groupby(["session_code", "round"]).apply(lambda x: pd.Series({
        "dom_share_initial": x.loc[x["is_dominant"] == 1, "initial_permits"].sum() / x["initial_permits"].sum(),
        "dom_share_efficient": x.loc[x["is_dominant"] == 1, "eff_holdings"].sum() / x["eff_holdings"].sum() if x["eff_holdings"].sum() > 0 else np.nan,
        "cap": x["initial_permits"].sum(), "sum_eff_holdings": x["eff_holdings"].sum(),
        "dominant_mc": x["group_dominant_mc"].iloc[0], "market_price": x["market_price"].iloc[0],
        "eq_price": x["eq_price"].iloc[0], "allocation": x["allocation"].iloc[0]})).reset_index()
    shares = sr.groupby(["dominant_mc", "market_price", "eq_price", "allocation"]).agg(
        dom_share_initial=("dom_share_initial", "mean"), dom_share_efficient=("dom_share_efficient", "mean"),
        cap_minus_sum_eff_holdings=("cap", lambda s: float((s - sr.loc[s.index, "sum_eff_holdings"]).mean()))).reset_index()
    det = theory_positions_det()
    out = pos.merge(shares, on=["dominant_mc", "market_price", "eq_price", "allocation"], how="left")
    out = out.merge(det, on=["dominant_mc", "market_price", "eq_price", "allocation"], how="left")
    out.to_csv(os.path.join(new_dir, "theory_positions_by_cell.csv"), index=False, encoding="utf-8-sig")
    coll = td.groupby(["allocation", "eq_price", "firm_type"]).agg(n=("initial_permits", "size"), initial_permits=("initial_permits", "mean"),
                                                                   eff_holdings=("eff_holdings", "mean"), theory_net=("theory_net", "mean"),
                                                                   actual_net_trade=("net_trade", "mean")).reset_index()
    coll.to_csv(os.path.join(new_dir, "theory_positions_by_alloc_price.csv"), index=False, encoding="utf-8-sig")
    hb = hahn_benchmark_table()
    hb.to_csv(os.path.join(new_dir, "hahn_benchmark_by_cell.csv"), index=False, encoding="utf-8-sig")
    hb.groupby(["allocation", "eq_price"]).agg(cells=("cap_det", "size"), mean_cap=("cap_det", "mean"),
                                               mean_comp_coalition_net_purchase=("comp_coalition_net_purchase", "mean"),
                                               mean_hahn_coalition_net_purchase=("coalition_net_purchase", "mean"),
                                               mean_hahn_price_minus_comp=("hahn_price_minus_comp", "mean"),
                                               cells_price_below_comp=("hahn_price_minus_comp", lambda s: int((s < 0).sum())),
                                               cells_price_above_comp=("hahn_price_minus_comp", lambda s: int((s > 0).sum())),
                                               mean_hahn_unused_permits=("hahn_unused_permits", "mean"),
                                               mean_surplus_loss_vs_comp=("hahn_surplus_loss_vs_comp", "mean")).reset_index().to_csv(
        os.path.join(new_dir, "hahn_benchmark_by_alloc_price.csv"), index=False, encoding="utf-8-sig")

    # ---- A6: volume gap ----
    _by_type_alloc_price(td, dict(n=("gap", "size"), actual_net_trade=("net_trade", "mean"), theory_net_trade=("theory_net", "mean"),
                                  mean_gap=("gap", "mean"), mean_abs_gap=("abs_gap", "mean"), mean_share_realised=("share_realised", "mean"),
                                  median_share_realised=("share_realised", "median"), wrong_direction_share=("wrong_direction", "mean"),
                                  n_with_theory_ge1=("share_realised", "count"))).to_csv(os.path.join(new_dir, "volume_gap_summary.csv"), index=False, encoding="utf-8-sig")
    f_gap = [f"gap ~ is_dominant*is_equal + {_EQ} + round", f"gap ~ is_dominant*is_equal*({_EQ}) + round",
             f"abs_gap ~ is_dominant*is_equal + {_EQ} + round", f"wrong_direction ~ is_dominant*is_equal + {_EQ} + round"]
    tidy = new_reg_table(td, f_gap, cluster_col, os.path.join(new_dir, "volume_gap_reg.txt"), "A6. Volume gap: actual minus competitive net trade (firm-round)",
                         base_notes + ["gap = net_trade - theory_net (permits; positive = bought more / sold less than the competitive benchmark);",
                                       "abs_gap = |gap|; wrong_direction = 1 if the firm traded against the sign of its benchmark position (|theory_net| >= 1)."] + _DOM_ALLOC_NOTES,
                         hypotheses={f_gap[0]: _DOM_ALLOC_HYP, f_gap[2]: _DOM_ALLOC_HYP, f_gap[3]: _DOM_ALLOC_HYP})
    tidy.insert(0, "block", "A6_volume_gap"); tidy_all.append(tidy)
    big = td.dropna(subset=["share_realised"])
    tidy = new_reg_table(big, [f"share_realised ~ is_dominant*is_equal + {_EQ} + round"], cluster_col, os.path.join(new_dir, "volume_gap_reg_share_realised.txt"),
                         "A6b. Share of the competitive net trade realised (firm-rounds with |theory_net| >= 1)",
                         base_notes + ["share_realised = net_trade / theory_net (1 = exactly the competitive volume; 0 = no trade; negative = wrong direction)."] + _DOM_ALLOC_NOTES,
                         hypotheses={f"share_realised ~ is_dominant*is_equal + {_EQ} + round": _DOM_ALLOC_HYP})
    tidy.insert(0, "block", "A6_share_realised"); tidy_all.append(tidy)

    # ---- A7: unused and constraining permits ----
    neg = int((td["unused"] < 0).sum())
    _by_type_alloc_price(td, dict(n=("unused", "size"), mean_unused=("unused", "mean"), share_unused_pos=("unused_pos", "mean"),
                                  mean_final_permits=("final_permits", "mean"), mean_emission=("emission", "mean"),
                                  share_constrained=("constrained", "mean"), mean_short_of_efficient=("short_of_efficient", "mean"))).assign(
        rows_with_negative_unused=neg).to_csv(os.path.join(new_dir, "unused_permits.csv"), index=False, encoding="utf-8-sig")
    rnd = td.groupby(["session_code", "round"]).apply(lambda x: pd.Series({
        "is_equal": x["is_equal"].iloc[0], "eq_price": x["eq_price"].iloc[0], "eq_price_low": x["eq_price_low"].iloc[0], "eq_price_high": x["eq_price_high"].iloc[0],
        "cap": x["initial_permits"].sum(), "emissions": x["emission"].sum(), "unused_total": x["unused"].sum(), "sum_net_trade": x["net_trade"].sum(),
        "unused_dominant": x.loc[x["is_dominant"] == 1, "unused"].sum(), "unused_small": x.loc[x["is_dominant"] == 0, "unused"].sum(),
        "n_constrained_dominant": x.loc[x["is_dominant"] == 1, "constrained"].sum(), "n_constrained_small": x.loc[x["is_dominant"] == 0, "constrained"].sum()})).reset_index()
    rnd["share_cap_unused"] = rnd["unused_total"] / rnd["cap"]
    rnd["check_cap_minus_emissions_equals_unused"] = (rnd["cap"] - rnd["emissions"] - rnd["unused_total"]).abs() < 1e-9   # fails only where sum_net_trade != 0
    rnd.to_csv(os.path.join(new_dir, "unused_permits_round.csv"), index=False, encoding="utf-8-sig")
    f_un = [f"unused ~ is_dominant*is_equal + {_EQ} + round", f"constrained ~ is_dominant*is_equal + {_EQ} + round",
            f"short_of_efficient ~ is_dominant*is_equal + {_EQ} + round"]
    tidy = new_reg_table(td, f_un, cluster_col, os.path.join(new_dir, "unused_permits_reg.txt"), "A7. Unused and constraining permits (firm-round)",
                         base_notes + ["unused = final_permits - emission (permits left over at the end of the round; final_permits = post-trading holdings);",
                                       "constrained = 1 if final_permits < competitive holdings at eq_price; short_of_efficient = max(0, competitive holdings - final_permits).",
                                       f"Rows with negative unused permits: {neg}."] + _DOM_ALLOC_NOTES, hypotheses={f: _DOM_ALLOC_HYP for f in f_un})
    tidy.insert(0, "block", "A7_unused"); tidy_all.append(tidy)
    f_rnd = [f"share_cap_unused ~ is_equal*({_EQ}) + round"]
    tidy = new_reg_table(rnd, f_rnd, cluster_col, os.path.join(new_dir, "unused_permits_round_reg.txt"), "A7b. Share of the cap left unused (session-round)",
                         ["share_cap_unused = (cap - total emissions) / cap, cap = sum of initial permits in the session-round."],
                         hypotheses={f_rnd[0]: ["is_equal = 0", "is_equal + is_equal:eq_price_low = 0", "is_equal + is_equal:eq_price_high = 0"]})
    tidy.insert(0, "block", "A7_unused_round"); tidy_all.append(tidy)

    # ---- A9: profit decomposition ----
    _by_type_alloc_price(td, dict(n=("net_profit", "size"), production_profit=("production_profit", "mean"), trading_cash_flow=("trading_cf", "mean"),
                                  net_profit=("net_profit", "mean"), bench_production_profit=("benchmark_production_profit", "mean"),
                                  bench_trading_cash_flow=("trading_cf_bench", "mean"), bench_net_profit=("net_profit_bench", "mean"),
                                  prod_gap=("prod_gap", "mean"), trade_gap=("trade_gap", "mean"), net_gap=("net_gap", "mean"))).to_csv(
        os.path.join(new_dir, "profit_decomposition.csv"), index=False, encoding="utf-8-sig")
    f_pr = [f"{y} ~ is_dominant*is_equal + {_EQ} + round" for y in ["prod_gap", "trade_gap", "net_gap"]]
    tidy = new_reg_table(td, f_pr, cluster_col, os.path.join(new_dir, "profit_decomposition_reg.txt"), "A9. Profit decomposition relative to the competitive benchmark (firm-round)",
                         base_notes + ["net_profit = production_profit + trading_cash_flow (identity net_profit = revenue - total_cost + permits sold - permits bought holds in the raw data);",
                                       "benchmarks: production profit at the competitive output for eq_price; trading cash flow = -eq_price x competitive net trade.",
                                       "prod_gap / trade_gap / net_gap = actual - benchmark (ECU)."] + _DOM_ALLOC_NOTES, hypotheses={f: _DOM_ALLOC_HYP for f in f_pr})
    tidy.insert(0, "block", "A9_profit"); tidy_all.append(tidy)

    # ---- A11 / A12: Tables 5-6 specifications with session clustering, full sample and rounds 7-12 ----
    spec_all = ["emission_diff ~ is_tax + is_trade + is_dominant + is_dominant:is_tax + is_dominant:is_trade",
                "production_profit_diff ~ is_tax + is_trade + is_dominant + is_dominant:is_tax + is_dominant:is_trade"]
    spec_tr = [f"emission_diff ~ is_equal*is_dominant + {_EQ}", f"production_profit_diff ~ is_equal*is_dominant + {_EQ}"]
    hyp_all = {f: ["is_dominant:is_tax = 0", "is_dominant:is_trade = 0", "is_dominant:is_tax - is_dominant:is_trade = 0"] for f in spec_all}
    hyp_tr = {f: _DOM_ALLOC_HYP for f in spec_tr}
    for label, dfa, dft in (("full_sample", df, td), ("rounds_7_12", df[df["round"] >= 7], td[td["round"] >= 7])):
        t1 = new_reg_table(dfa, spec_all, cluster_col, os.path.join(new_dir, f"robustness_subj_tables5_6_{label}.txt"),
                           f"A11/A12. Table 5-6 specifications (all stages), session-clustered, {label}",
                           ["emission_diff = emission - e_soc; production_profit_diff = (revenue - total_cost) - pi_soc, as in the paper's Tables 5-6.",
                            "The paper's tables cluster by participant; here clusters are sessions (10) with wild-cluster-bootstrap p-values."], hypotheses=hyp_all)
        t1.insert(0, "block", f"A11_subj_all_{label}"); tidy_all.append(t1)
        t2 = new_reg_table(dft, spec_tr, cluster_col, os.path.join(new_dir, f"robustness_subj_allocation_{label}.txt"),
                           f"A11/A12. Table 5-6 allocation specifications (trading rounds), session-clustered, {label}", _DOM_ALLOC_NOTES, hypotheses=hyp_tr)
        t2.insert(0, "block", f"A11_subj_trade_{label}"); tidy_all.append(t2)

    pd.concat(tidy_all, ignore_index=True).to_csv(os.path.join(new_dir, "wildboot_pvalues_subject.csv"), index=False, encoding="utf-8-sig")
    print(f"[NEW] subject-level analyses written to {new_dir}")
    return td
# --- END NEW ---

# --- NEW: allocation/Hahn framing --- run the additional subject-level analyses
if RUN_NEW_ANALYSES:
    run_new_subject_analyses(df, outcome_samples["net_trade"], OUTPUT_DIR, cluster_col="session_code")
# --- END NEW ---
