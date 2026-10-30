import ast

import numpy as np
import pandas as pd


SOCIAL_COST_PER_EMISSION = 6.0
TRADE_EQ_TAX_LOW = 3.0
TRADE_EQ_TAX_BASE = 6.0
TRADE_EQ_TAX_HIGH = 9.0


def _as_numeric(series, default=0.0):
    if np.isscalar(series):
        return pd.to_numeric(pd.Series([series]), errors="coerce").fillna(default).iloc[0]
    return pd.to_numeric(series, errors="coerce").fillna(default)


def infer_equivalent_tax_rate(df):
    is_trade = _as_numeric(df.get("is_trade", 0))
    is_tax = _as_numeric(df.get("is_tax", 0))
    eq_price_low = _as_numeric(df.get("eq_price_low", 0))
    eq_price_high = _as_numeric(df.get("eq_price_high", 0))

    indicator_tax_rate = np.where(
        eq_price_low == 1,
        TRADE_EQ_TAX_LOW,
        np.where(eq_price_high == 1, TRADE_EQ_TAX_HIGH, TRADE_EQ_TAX_BASE),
    )

    trade_rate = np.where(
        eq_price_low == 1,
        TRADE_EQ_TAX_LOW,
        np.where(eq_price_high == 1, TRADE_EQ_TAX_HIGH, TRADE_EQ_TAX_BASE),
    )
    return np.where(is_tax == 1, indicator_tax_rate, np.where(is_trade == 1, trade_rate, 0.0))


def _parse_disturbance_values(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    if isinstance(value, str):
        return ast.literal_eval(value)
    return []


def compute_subject_theory_benchmark(df):
    out = df.copy()

    market_price = _as_numeric(out["market_price"])
    mc = _as_numeric(out["marginal_cost_coefficient"])
    emission_per_unit = _as_numeric(out["carbon_emission_per_unit"])
    max_production = _as_numeric(out["max_production"])
    equivalent_tax_rate = infer_equivalent_tax_rate(out)

    disturbances = out["disturbance_values"].apply(_parse_disturbance_values)

    theoretical_q = []
    theoretical_total_cost = []
    theoretical_policy_payment = []
    theoretical_net_profit = []

    for price_i, mc_i, epu_i, qmax_i, tax_i, vals in zip(
        market_price, mc, emission_per_unit, max_production, equivalent_tax_rate, disturbances
    ):
        best_q = 0
        best_profit = 0.0
        best_cost = 0.0
        best_payment = 0.0
        cumulative_disturbance = 0.0

        for q_i in range(1, int(qmax_i) + 1):
            cumulative_disturbance += float(vals[q_i - 1])
            total_cost_i = float(mc_i) * q_i * (q_i + 1) / 2 + cumulative_disturbance
            policy_payment_i = float(tax_i) * float(epu_i) * q_i
            profit_i = float(price_i) * q_i - total_cost_i - policy_payment_i
            if profit_i > best_profit + 1e-9:
                best_q = q_i
                best_profit = profit_i
                best_cost = total_cost_i
                best_payment = policy_payment_i

        theoretical_q.append(best_q)
        theoretical_total_cost.append(best_cost)
        theoretical_policy_payment.append(best_payment)
        theoretical_net_profit.append(best_profit)

    theoretical_q = pd.Series(theoretical_q, index=out.index, dtype=float)
    theoretical_total_cost = pd.Series(theoretical_total_cost, index=out.index, dtype=float)
    theoretical_policy_payment = pd.Series(theoretical_policy_payment, index=out.index, dtype=float)
    theoretical_net_profit = pd.Series(theoretical_net_profit, index=out.index, dtype=float)
    theoretical_emission = theoretical_q * emission_per_unit
    theoretical_revenue = market_price * theoretical_q
    theoretical_surplus = (
        theoretical_net_profit
        + theoretical_policy_payment
        - SOCIAL_COST_PER_EMISSION * theoretical_emission
    )

    out["benchmark_equivalent_tax_rate"] = equivalent_tax_rate
    out["benchmark_production"] = theoretical_q.astype(float)
    out["benchmark_emission"] = theoretical_emission.astype(float)
    out["benchmark_final_permits"] = theoretical_emission.astype(float)
    out["benchmark_total_cost"] = theoretical_total_cost
    out["benchmark_policy_payment"] = theoretical_policy_payment.astype(float)
    out["benchmark_net_profit"] = theoretical_net_profit.astype(float)
    out["benchmark_surplus"] = theoretical_surplus.astype(float)
    if "initial_permits" in out.columns:
        initial_permits = pd.to_numeric(out["initial_permits"], errors="coerce")
        out["benchmark_net_trade"] = out["benchmark_final_permits"] - initial_permits
    return out


def aggregate_subject_benchmark_to_group(subj_df):
    keys = ["session_code", "app", "round"]
    return (
        subj_df.groupby(keys, dropna=False)
        .agg(
            benchmark_production=("benchmark_production", "sum"),
            benchmark_emission=("benchmark_emission", "sum"),
            benchmark_net_profit=("benchmark_net_profit", "sum"),
            benchmark_surplus=("benchmark_surplus", "sum"),
        )
        .reset_index()
    )


# --- NEW: allocation/Hahn framing ---
# Deterministic (epsilon = 0) theory objects per parameter cell: efficient holdings, initial
# allocations, net positions, and a Hahn (1984) dominant-coalition manipulation benchmark.
N_DOMINANT, N_SMALL = 3, 12
CAP_DOMINANT, CAP_SMALL = 20, 8
EMIS_DOMINANT, EMIS_SMALL = 2, 1
GRANDFATHER_DOMINANT_SHARE = 0.40
COST_PAIRS = ((3, 4), (6, 8))   # (dominant a, small a) as implemented in oTree; the draft text says (3,6)/(4,8) -- see README
CELL_GRID = [(a_d, a_s, p, c) for (a_d, a_s) in COST_PAIRS for p in (25, 35) for c in (3, 6, 9)]
HAHN_PRICE_GRID = list(range(0, 61))


def det_best_q(price, a, emis_per_unit, permit_price, qmax):
    """Game rule without cost shocks: produce every unit with price > a*q + emis_per_unit*permit_price."""
    q = 0
    for qi in range(1, int(qmax) + 1):
        if price > a * qi + emis_per_unit * permit_price:
            q = qi
    return q


def det_production_profit(price, a, q):
    """Revenue minus cumulative marginal cost a*1 + ... + a*q."""
    return price * q - a * q * (q + 1) / 2.0


def det_cell_benchmark(a_d, a_s, price, eq_price):
    """Competitive benchmark of one cell: outputs, efficient holdings, cap, and nominal allocations."""
    q_d = det_best_q(price, a_d, EMIS_DOMINANT, eq_price, CAP_DOMINANT)
    q_s = det_best_q(price, a_s, EMIS_SMALL, eq_price, CAP_SMALL)
    e_d, e_s = EMIS_DOMINANT * q_d, EMIS_SMALL * q_s
    cap = N_DOMINANT * e_d + N_SMALL * e_s
    alloc = {
        "grandfathering": (GRANDFATHER_DOMINANT_SHARE * cap / N_DOMINANT, (1 - GRANDFATHER_DOMINANT_SHARE) * cap / N_SMALL),
        "equal": (cap / (N_DOMINANT + N_SMALL), cap / (N_DOMINANT + N_SMALL)),
    }
    return dict(a_d=a_d, a_s=a_s, price=price, eq_price=eq_price, q_d=q_d, q_s=q_s, e_d=e_d, e_s=e_s, cap=cap, alloc=alloc)


def theory_positions_det():
    """Cell x allocation table of efficient holdings, nominal initial permits and net positions (epsilon = 0)."""
    rows = []
    for a_d, a_s, price, c in CELL_GRID:
        b = det_cell_benchmark(a_d, a_s, price, c)
        for alloc, (perm_d, perm_s) in b["alloc"].items():
            rows.append({
                "dominant_mc": a_d, "small_mc": b["a_s"], "market_price": price, "eq_price": c, "allocation": alloc,
                "cap_det": b["cap"], "dom_eff_output_det": b["q_d"], "small_eff_output_det": b["q_s"],
                "dom_eff_holdings_det": b["e_d"], "small_eff_holdings_det": b["e_s"],
                "dom_share_of_cap_det": (N_DOMINANT * b["e_d"] / b["cap"]) if b["cap"] else np.nan,
                "dom_initial_permits_nominal": perm_d, "small_initial_permits_nominal": perm_s,
                "dom_net_position_det": b["e_d"] - perm_d, "small_net_position_det": b["e_s"] - perm_s,
                "dom_position_label": "net buyer" if b["e_d"] - perm_d > 0.5 else ("net seller" if b["e_d"] - perm_d < -0.5 else "about efficient"),
            })
    return pd.DataFrame(rows)


def hahn_coalition_benchmark(a_d, a_s, price, eq_price, allocation, price_grid=HAHN_PRICE_GRID):
    """Hahn (1984) benchmark with the three dominant firms acting as one price-setting coalition.

    Fringe: 12 identical price-taking small firms; at permit price P each produces det_best_q(price, a_s, 1, P, 8)
    and offers (initial_s - q_s) permits. The coalition picks its net purchase x (integer) and pays the lowest
    grid price at which the fringe supplies x (or, when selling, receives the highest price at which the fringe
    demands |x|). Coalition permits are used in whole dominant units (2 permits each), never beyond the
    unregulated optimum. Returns the manipulated price, quantities, and surplus, next to the competitive values.
    """
    b = det_cell_benchmark(a_d, a_s, price, eq_price)
    perm_d, perm_s = b["alloc"][allocation]
    q_mkt_d = det_best_q(price, a_d, EMIS_DOMINANT, 0, CAP_DOMINANT)

    def fringe_net_supply(P):
        return N_SMALL * (perm_s - det_best_q(price, a_s, EMIS_SMALL, P, CAP_SMALL))

    def coalition_profit(holdings):
        q_each = min(q_mkt_d, int(np.floor(holdings / N_DOMINANT / EMIS_DOMINANT)))
        return N_DOMINANT * det_production_profit(price, a_d, q_each), q_each

    supply = {P: fringe_net_supply(P) for P in price_grid}
    best = None
    x_min = -int(np.floor(N_DOMINANT * perm_d))
    x_max = int(np.floor(N_SMALL * perm_s))
    for x in range(x_min, x_max + 1):
        if x > 0:
            feas = [P for P in price_grid if supply[P] >= x]
            if not feas:
                continue
            P = min(feas)
        elif x < 0:
            feas = [P for P in price_grid if -supply[P] >= -x]
            if not feas:
                continue
            P = max(feas)
        else:
            P = eq_price
        prof, q_each = coalition_profit(N_DOMINANT * perm_d + x)
        total = prof - P * x
        if best is None or total > best["coalition_profit"] + 1e-9:
            q_s = det_best_q(price, a_s, EMIS_SMALL, P, CAP_SMALL)
            best = dict(hahn_price=P, coalition_net_purchase=x, dom_output_each=q_each, small_output_each=q_s,
                        coalition_profit=total, fringe_profit=N_SMALL * det_production_profit(price, a_s, q_s),
                        emissions_total=N_DOMINANT * EMIS_DOMINANT * q_each + N_SMALL * q_s)
    # competitive reference (same discrete model at P = eq_price)
    prof_c, q_each_c = coalition_profit(N_DOMINANT * b["e_d"])
    comp = dict(comp_price=eq_price, comp_dom_output_each=b["q_d"], comp_small_output_each=b["q_s"],
                comp_coalition_net_purchase=N_DOMINANT * (b["e_d"] - perm_d),
                comp_emissions_total=b["cap"],
                comp_surplus=N_DOMINANT * det_production_profit(price, a_d, b["q_d"]) + N_SMALL * det_production_profit(price, a_s, b["q_s"]) - SOCIAL_COST_PER_EMISSION * b["cap"])
    out = dict(dominant_mc=a_d, small_mc=a_s, market_price=price, eq_price=eq_price, allocation=allocation,
               cap_det=b["cap"], dom_initial_permits_nominal=perm_d, small_initial_permits_nominal=perm_s)
    out.update(comp)
    out.update(best)
    out["hahn_surplus"] = best["coalition_profit"] + best["hahn_price"] * best["coalition_net_purchase"] + best["fringe_profit"] - SOCIAL_COST_PER_EMISSION * best["emissions_total"]
    out["hahn_price_minus_comp"] = best["hahn_price"] - eq_price
    out["hahn_unused_permits"] = b["cap"] - best["emissions_total"]
    out["hahn_surplus_loss_vs_comp"] = comp["comp_surplus"] - out["hahn_surplus"]
    return out


def hahn_benchmark_table():
    rows = []
    for a_d, a_s, price, c in CELL_GRID:
        for allocation in ("grandfathering", "equal"):
            rows.append(hahn_coalition_benchmark(a_d, a_s, price, c, allocation))
    return pd.DataFrame(rows)
# --- END NEW ---
