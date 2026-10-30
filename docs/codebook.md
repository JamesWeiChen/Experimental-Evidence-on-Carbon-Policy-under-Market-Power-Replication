# Codebook: constructed analysis files

Auto-generated summary of the four panels written by `build/code/` (identical copies are written to `build/output/` and `analysis/input/`). Generated with pandas from the files produced by `run_all.py`.


## `subj_lv_panel.csv`

Subject x stage x round panel (one row per firm-round; includes post-experiment survey answers). Rows: 5,400. Columns: 84.

| Variable | Type | Non-missing | Min | Max |
|---|---|---|---|---|
| `participant_code` | object | 5,400 |  |  |
| `session_code` | object | 5,400 |  |  |
| `subject_id` | int64 | 5,400 | 1 | 15 |
| `app` | object | 5,400 |  |  |
| `round` | int64 | 5,400 | 1 | 12 |
| `is_tax` | int64 | 5,400 | 0 | 1 |
| `is_trade` | int64 | 5,400 | 0 | 1 |
| `is_equal` | int64 | 5,400 | 0 | 1 |
| `tax_rate_low` | int64 | 5,400 | 0 | 1 |
| `tax_rate_high` | int64 | 5,400 | 0 | 1 |
| `eq_price_low` | int64 | 5,400 | 0 | 1 |
| `eq_price_high` | int64 | 5,400 | 0 | 1 |
| `age` | int64 | 5,400 | 18 | 44 |
| `carbon_emission_per_unit` | int64 | 5,400 | 1 | 2 |
| `carbon_tax_efficiency` | int64 | 5,400 | 1 | 5 |
| `carbon_tax_environment` | int64 | 5,400 | 1 | 5 |
| `carbon_tax_fairness` | int64 | 5,400 | 1 | 5 |
| `carbon_tax_paid` | float64 | 5,400 | 0 | 270 |
| `carbon_trade_efficiency` | int64 | 5,400 | 1 | 5 |
| `carbon_trade_environment` | int64 | 5,400 | 1 | 5 |
| `carbon_trade_fairness` | int64 | 5,400 | 1 | 5 |
| `carbon_trade_mkt_power` | int64 | 5,400 | 1 | 5 |
| `disturbance_values` | object | 5,400 |  |  |
| `e_mkt` | float64 | 5,400 | 3 | 22 |
| `e_policy` | float64 | 5,400 | 1 | 22 |
| `e_soc` | float64 | 5,400 | 2 | 14 |
| `e_tax` | float64 | 5,400 | 1 | 22 |
| `emission` | int64 | 5,400 | 0 | 30 |
| `final_cash` | float64 | 5,400 | -161 | 1900 |
| `final_permits` | float64 | 1,800 | 0 | 25 |
| `grade` | object | 5,400 |  |  |
| `group_E_mkt` | float64 | 5,400 | 58 | 162 |
| `group_E_policy` | float64 | 5,400 | 20 | 162 |
| `group_E_soc` | float64 | 5,400 | 34 | 126 |
| `group_E_tax` | float64 | 5,400 | 20 | 162 |
| `group_Pi_mkt` | float64 | 5,400 | 430 | 2209 |
| `group_Pi_policy` | float64 | 5,400 | 98 | 2209 |
| `group_Pi_soc` | float64 | 5,400 | 399 | 2095 |
| `group_Pi_tax` | float64 | 5,400 | 98 | 2209 |
| `group_Q_mkt` | float64 | 5,400 | 47 | 129 |
| `group_Q_policy` | float64 | 5,400 | 17 | 129 |
| `group_Q_soc` | float64 | 5,400 | 29 | 105 |
| `group_Q_tax` | float64 | 5,400 | 17 | 129 |
| `group_carbon_multiplier` | float64 | 5,400 | 0.5 | 1.5 |
| `group_dominant_mc` | int64 | 5,400 | 3 | 6 |
| `group_emission` | int64 | 5,400 | 14 | 161 |
| `group_market_price` | int64 | 5,400 | 25 | 35 |
| `group_non_dominant_mc` | int64 | 5,400 | 4 | 8 |
| `group_tax_rate` | int64 | 5,400 | 3 | 9 |
| `has_env_econ` | int64 | 5,400 | 0 | 1 |
| `has_exp_econ` | int64 | 5,400 | 0 | 1 |
| `has_intro_econ` | int64 | 5,400 | 0 | 1 |
| `has_micro` | int64 | 5,400 | 0 | 1 |
| `has_pub_econ` | int64 | 5,400 | 0 | 1 |
| `id_in_group` | int64 | 5,400 | 1 | 15 |
| `initial_capital` | float64 | 5,400 | 0 | 1000 |
| `initial_permits` | float64 | 1,800 | 1 | 20 |
| `is_dominant` | int64 | 5,400 | 0 | 1 |
| `main_goal` | object | 5,400 |  |  |
| `major_econ_or_bz` | int64 | 5,400 | 0 | 1 |
| `major_env` | int64 | 5,400 | 0 | 1 |
| `male` | int64 | 5,400 | 0 | 1 |
| `marginal_cost_coefficient` | int64 | 5,400 | 3 | 8 |
| `market_price` | float64 | 5,400 | 25 | 35 |
| `max_production` | int64 | 5,400 | 8 | 20 |
| `net_profit` | float64 | 5,400 | -894 | 900 |
| `net_trade` | float64 | 1,800 | -9 | 17 |
| `pi_mkt` | float64 | 5,400 | 24 | 192 |
| `pi_policy` | float64 | 5,400 | 0 | 191 |
| `pi_soc` | float64 | 5,400 | 20 | 166 |
| `pi_tax` | float64 | 5,400 | 0 | 192 |
| `production` | int64 | 5,400 | 0 | 15 |
| `q_mkt` | float64 | 5,400 | 3 | 11 |
| `q_policy` | float64 | 5,400 | 1 | 11 |
| `q_soc` | float64 | 5,400 | 1 | 7 |
| `q_tax` | float64 | 5,400 | 1 | 11 |
| `respond_to_high_others` | int64 | 5,400 | -1 | 1 |
| `revenue` | float64 | 5,400 | 0 | 525 |
| `selected_round` | int64 | 5,400 | 1 | 11 |
| `total_cost` | float64 | 5,400 | 0 | 363.9 |
| `surplus` | float64 | 5,400 | -966 | 900 |
| `surplus_soc` | float64 | 5,400 | 6 | 95 |
| `group_surplus` | float64 | 5,400 | -156 | 1322 |
| `group_surplus_soc` | float64 | 5,400 | 183 | 1339 |

## `group_lv_panel.csv`

Market (session) x stage x round panel. Rows: 5,400. Columns: 32.

| Variable | Type | Non-missing | Min | Max |
|---|---|---|---|---|
| `session_code` | object | 5,400 |  |  |
| `app` | object | 5,400 |  |  |
| `round` | int64 | 5,400 | 1 | 12 |
| `is_tax` | int64 | 5,400 | 0 | 1 |
| `is_trade` | int64 | 5,400 | 0 | 1 |
| `is_equal` | int64 | 5,400 | 0 | 1 |
| `tax_rate_low` | int64 | 5,400 | 0 | 1 |
| `tax_rate_high` | int64 | 5,400 | 0 | 1 |
| `eq_price_low` | int64 | 5,400 | 0 | 1 |
| `eq_price_high` | int64 | 5,400 | 0 | 1 |
| `E_mkt` | float64 | 5,400 | 58 | 162 |
| `E_policy` | float64 | 5,400 | 20 | 162 |
| `E_soc` | float64 | 5,400 | 34 | 126 |
| `E_tax` | float64 | 5,400 | 20 | 162 |
| `Pi_mkt` | float64 | 5,400 | 430 | 2209 |
| `Pi_policy` | float64 | 5,400 | 98 | 2209 |
| `Pi_soc` | float64 | 5,400 | 399 | 2095 |
| `Pi_tax` | float64 | 5,400 | 98 | 2209 |
| `Q_mkt` | float64 | 5,400 | 47 | 129 |
| `Q_policy` | float64 | 5,400 | 17 | 129 |
| `Q_soc` | float64 | 5,400 | 29 | 105 |
| `Q_tax` | float64 | 5,400 | 17 | 129 |
| `carbon_multiplier` | float64 | 5,400 | 0.5 | 1.5 |
| `dominant_mc` | int64 | 5,400 | 3 | 6 |
| `emission` | int64 | 5,400 | 14 | 161 |
| `market_price` | int64 | 5,400 | 25 | 35 |
| `non_dominant_mc` | int64 | 5,400 | 4 | 8 |
| `tax_rate` | int64 | 5,400 | 3 | 9 |
| `surplus` | float64 | 5,400 | -156 | 1322 |
| `surplus_soc` | float64 | 5,400 | 183 | 1339 |
| `net_profit` | float64 | 5,400 | -69 | 2162 |
| `production` | int64 | 5,400 | 14 | 127 |

## `trades_panel.csv`

One row per executed permit trade in the Carbon Trading stage. Rows: 1,863. Columns: 14.

| Variable | Type | Non-missing | Min | Max |
|---|---|---|---|---|
| `session_code` | object | 1,863 |  |  |
| `round` | int64 | 1,863 | 1 | 12 |
| `trade_index` | int64 | 1,863 | 1 | 30 |
| `seconds` | int64 | 1,863 | 4 | 123 |
| `buyer_id` | int64 | 1,863 | 1 | 15 |
| `seller_id` | int64 | 1,863 | 1 | 15 |
| `price` | int64 | 1,863 | 1 | 250 |
| `quantity` | int64 | 1,863 | 1 | 19 |
| `is_equal` | int64 | 1,863 | 0 | 1 |
| `buyer_is_dominant` | int64 | 1,863 | 0 | 1 |
| `seller_is_dominant` | int64 | 1,863 | 0 | 1 |
| `eq_price` | float64 | 1,863 | 3 | 9 |
| `eq_price_low` | int64 | 1,863 | 0 | 1 |
| `eq_price_high` | int64 | 1,863 | 0 | 1 |

## `offers_panel.csv`

One row per bid/ask submitted in the Carbon Trading stage. Rows: 8,070. Columns: 13.

| Variable | Type | Non-missing | Min | Max |
|---|---|---|---|---|
| `session_code` | object | 8,070 |  |  |
| `round` | int64 | 8,070 | 1 | 12 |
| `id_in_group` | int64 | 8,070 | 1 | 15 |
| `is_dominant` | int64 | 8,070 | 0 | 1 |
| `offer_index` | int64 | 8,070 | 1 | 43 |
| `seconds` | int64 | 8,070 | 2 | 127 |
| `direction` | object | 8,070 |  |  |
| `price` | object | 8,070 |  |  |
| `quantity` | int64 | 8,070 | 1 | 1000 |
| `is_equal` | int64 | 8,070 | 0 | 1 |
| `eq_price` | float64 | 8,070 | 3 | 9 |
| `eq_price_low` | int64 | 8,070 | 0 | 1 |
| `eq_price_high` | int64 | 8,070 | 0 | 1 |
