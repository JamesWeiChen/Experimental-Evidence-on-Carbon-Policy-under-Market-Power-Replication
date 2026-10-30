"""Collapse the A7/A8 outputs into the exact cells reported in two manuscript tables.

Added for the replication package: these two tables were previously assembled by
hand from CSV outputs written by run_subj_lv_reg.py and run_group_lv_reg.py.
Run after those scripts.

Outputs (analysis/output/new/):
  table_welfare_decomposition_panelA.csv  -> Table "welfare_decomposition", Panel A
  table_permit_shortfalls_unused.csv      -> Table "permit_shortfalls_unused"
"""
import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NEW_DIR = os.path.join(BASE_DIR, "..", "output", "new")


def eq_price(df):
    return df["eq_price_low"].map({1: 3}).fillna(df["eq_price_high"].map({1: 9, 0: 6})).astype(int)


# Panel A: unadjusted market-round means of surplus and the two loss components, by policy and matched price.
eff = pd.read_csv(os.path.join(NEW_DIR, "efficiency_decomposition.csv"))
eff["matched_carbon_price"] = eq_price(eff)
eff["policy"] = eff["app"].map({"Stage_CarbonTax": "tax", "Stage_CarbonTrading": "trade"})
panel_a = (eff.groupby(["matched_carbon_price", "policy"])[["surplus_actual", "cross_firm_loss", "aggregate_wedge_surplus"]]
           .mean().unstack("policy"))
panel_a.columns = [f"{p}_{v}" for v, p in panel_a.columns]
panel_a = panel_a[["tax_surplus_actual", "trade_surplus_actual", "tax_cross_firm_loss", "trade_cross_firm_loss",
                   "tax_aggregate_wedge_surplus", "trade_aggregate_wedge_surplus"]].round(2).reset_index()
panel_a.to_csv(os.path.join(NEW_DIR, "table_welfare_decomposition_panelA.csv"), index=False, encoding="utf-8-sig")

# Permit shortfalls (dominant firms, firm-round) and unused permits (market-round), by allocation and matched price.
firm = pd.read_csv(os.path.join(NEW_DIR, "unused_permits.csv"))
dom = firm[firm["firm_type"] == "dominant"].set_index(["allocation", "eq_price"])
rnd = pd.read_csv(os.path.join(NEW_DIR, "unused_permits_round.csv"))
rnd["allocation"] = rnd["is_equal"].map({0: "grandfathering", 1: "equal"})
mkt = rnd.groupby(["allocation", "eq_price"]).agg(unused_per_market=("unused_total", "mean"),
                                                  unused_held_by_small=("unused_small", "mean"),
                                                  unused_pct_of_cap=("share_cap_unused", "mean"))
mkt["unused_pct_of_cap"] *= 100
permits = pd.DataFrame({
    "dominant_with_shortfall_pct": (dom["share_constrained"] * 100).round(1),
    "avg_dominant_shortfall": dom["mean_short_of_efficient"].round(2),
}).join(mkt.round(2)).reset_index()
permits["allocation"] = pd.Categorical(permits["allocation"], ["grandfathering", "equal"], ordered=True)
permits = permits.sort_values(["allocation", "eq_price"])
permits.to_csv(os.path.join(NEW_DIR, "table_permit_shortfalls_unused.csv"), index=False, encoding="utf-8-sig")

print(panel_a.to_string(index=False))
print()
print(permits.to_string(index=False))
