import pandas as pd
import numpy as np
import os
import glob
import re

# ==============================================================================
# Part 0: Setup File Paths and Directories
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'input')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'output')
OUTPUT_FILENAME = 'subj_lv_panel.csv'
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_file_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
print(f"腳本位置: {SCRIPT_DIR}")
print(f"Input 資料夾路徑: {INPUT_DIR}")
print(f"Output 資料夾路徑: {OUTPUT_DIR}")


# ==============================================================================
# Part 1: Load and Combine Data
# ==============================================================================
all_csv_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
if not all_csv_files:
    print(f"\n錯誤：在 '{INPUT_DIR}' 資料夾中找不到任何 CSV 檔案。")
    exit()
print(f"\n找到 {len(all_csv_files)} 個 CSV 檔案，將進行合併...")
list_of_dfs = [pd.read_csv(f) for f in all_csv_files]
df_wide = pd.concat(list_of_dfs, axis=0, ignore_index=True)
print(f"檔案合併完成。合併後的資料共有 {df_wide.shape[0]} 行。")


# ==============================================================================
# Part 2: Reshape Data to Panel Format
# ==============================================================================
id_vars = ['participant.code', 'session.code', 'participant.id_in_session']
static_cols_full_names = [
    'Stage_Survey.1.player.age', 'Stage_Survey.1.player.male', 'Stage_Survey.1.player.grade',
    'Stage_Survey.1.player.major_econ_or_bz', 'Stage_Survey.1.player.major_env',
    'Stage_Survey.1.player.has_intro_econ', 'Stage_Survey.1.player.has_micro',
    'Stage_Survey.1.player.has_env_econ', 'Stage_Survey.1.player.has_pub_econ',
    'Stage_Survey.1.player.has_exp_econ', 'Stage_Survey.1.player.main_goal',
    'Stage_Survey.1.player.respond_to_high_others', 'Stage_Survey.1.player.carbon_tax_fairness',
    'Stage_Survey.1.player.carbon_tax_efficiency', 'Stage_Survey.1.player.carbon_tax_environment',
    'Stage_Survey.1.player.carbon_trade_fairness', 'Stage_Survey.1.player.carbon_trade_efficiency',
    'Stage_Survey.1.player.carbon_trade_environment', 'Stage_Survey.1.player.carbon_trade_mkt_power'
]
static_cols_map = {col: col.split('.')[-1] for col in static_cols_full_names}
static_source_cols = id_vars + static_cols_full_names
if 'session.config.allocation_method' in df_wide.columns:
    static_source_cols.append('session.config.allocation_method')
existing_static_cols = [col for col in static_source_cols if col in df_wide.columns]
static_df = df_wide[existing_static_cols].copy()
static_df.rename(columns=static_cols_map, inplace=True)
static_df.rename(columns={'participant.code': 'participant_code', 'session.code': 'session_code', 'participant.id_in_session': 'subject_id'}, inplace=True)

apps_to_process = ['Stage_Control', 'Stage_CarbonTax', 'Stage_CarbonTrading']
cols_to_melt = [col for col in df_wide.columns if any(app_name in col for app_name in apps_to_process)]
df_melted = df_wide.melt(
    id_vars=id_vars, value_vars=cols_to_melt, var_name='original_col', value_name='value'
)
df_melted.dropna(subset=['value'], inplace=True)

def parse_otree_col(col_name):
    parts = col_name.split('.')
    if len(parts) < 4: return None, None, None
    app, round_num, obj_type = parts[0], parts[1], parts[2]
    var_base_name = '.'.join(parts[3:])
    if obj_type == 'player': new_var_name = var_base_name
    elif obj_type in ['group', 'subsession']: new_var_name = f'group_{var_base_name}'
    else: return None, None, None
    try: round_num_int = int(round_num)
    except ValueError: return None, None, None
    return app, round_num_int, new_var_name

parsed_cols = df_melted['original_col'].apply(lambda x: pd.Series(parse_otree_col(x)))
parsed_cols.columns = ['app', 'round', 'variable']
df_melted = pd.concat([df_melted, parsed_cols], axis=1)
df_melted.dropna(subset=['app', 'round', 'variable'], inplace=True)
df_melted['round'] = df_melted['round'].astype(int)

df_panel = df_melted.pivot_table(
    index=['participant.code', 'session.code', 'participant.id_in_session', 'app', 'round'],
    columns='variable', values='value', aggfunc='first'
).reset_index()

df_panel.rename(columns={'participant.code': 'participant_code', 'session.code': 'session_code', 'participant.id_in_session': 'subject_id'}, inplace=True)
df_final = pd.merge(df_panel, static_df, on=['participant_code', 'session_code', 'subject_id'], how='left')


# ==============================================================================
# Part 3: Generate New Dummy Variables
# ==============================================================================
print("\n開始新增指定的虛擬變數...")
df_final['is_tax'] = (df_final['app'] == 'Stage_CarbonTax').astype(int)
df_final['is_trade'] = (df_final['app'] == 'Stage_CarbonTrading').astype(int)
if 'session.config.allocation_method' in df_final.columns:
    df_final['is_equal'] = (df_final['session.config.allocation_method'] == 'equal').astype(int)
    df_final.drop(columns=['session.config.allocation_method'], inplace=True)
else: df_final['is_equal'] = 0
if 'group_tax_rate' in df_final.columns:
    df_final['group_tax_rate'] = df_final['group_tax_rate'].fillna(0)
    df_final['tax_rate_low'] = (pd.to_numeric(df_final['group_tax_rate'], errors='coerce') == 3).astype(int)
    df_final['tax_rate_high'] = (pd.to_numeric(df_final['group_tax_rate'], errors='coerce') == 9).astype(int)
else:
    df_final['tax_rate_low'] = 0; df_final['tax_rate_high'] = 0
if 'group_carbon_multiplier' in df_final.columns:
    df_final['group_carbon_multiplier'] = df_final['group_carbon_multiplier'].fillna(0)
    df_final['eq_price_low'] = (pd.to_numeric(df_final['group_carbon_multiplier'], errors='coerce') == 0.5).astype(int)
    df_final['eq_price_high'] = (pd.to_numeric(df_final['group_carbon_multiplier'], errors='coerce') == 1.5).astype(int)
else:
    df_final['eq_price_low'] = 0; df_final['eq_price_high'] = 0
print("虛擬變數新增完成。")


# ==============================================================================
# Part 4: Final Data Cleaning
# ==============================================================================
print("\n開始進行最終資料清理...")
print(f"\n[診斷] 清理前的所有欄位名稱 (共 {len(df_final.columns)} 個): \n{df_final.columns.tolist()}")

# --- Step 4.0: Preserve permit holdings and compute net trade for trading rounds ---
if 'permits' in df_final.columns and 'current_permits' in df_final.columns:
    df_final['initial_permits'] = pd.to_numeric(df_final['permits'], errors='coerce')
    df_final['final_permits'] = pd.to_numeric(df_final['current_permits'], errors='coerce')
    df_final['net_trade'] = df_final['final_permits'] - df_final['initial_permits']
    if 'is_trade' in df_final.columns:
        non_trade_mask = df_final['is_trade'] != 1
        df_final.loc[non_trade_mask, ['initial_permits', 'final_permits', 'net_trade']] = np.nan
    print("\n[成功] 已保留 permit holdings，並新增 'initial_permits'、'final_permits'、'net_trade'。")
else:
    print("\n[警告] 找不到 'permits' 或 'current_permits' 欄位，無法計算 net_trade。")

# --- Step 4.1: Fill missing carbon_tax_paid with 0 ---
if 'carbon_tax_paid' in df_final.columns:
    df_final['carbon_tax_paid'] = df_final['carbon_tax_paid'].fillna(0)
    print("\n[成功] 已將 'carbon_tax_paid' 的缺失值補 0。")
else:
    print("\n[警告] 在資料中找不到 'carbon_tax_paid' 欄位，無法進行補 0 操作。")

# --- Step 4.2: Drop specified unnecessary columns ---
cols_to_drop = [
    'group_allocation_details', 'group_allocation_method', 'group_buy_orders',
    'group_cap_total', 'group_executed_trades', 'group_id_in_subsession',
    'group_price_history', 'group_round_number', 'group_sell_orders',
    'group_start_time', 'group_total_optimal_emissions', 'mkt_emissions',
    'mkt_production', 'optimal_emissions', 'optimal_production', 'permits',
    'submitted_offers', 'total_bought', 'total_earned', 'total_sold', 'total_spent',
    'current_cash', 'current_permits',
    'payoff'  # Added payoff to the drop list
]
existing_cols_to_drop = [col for col in cols_to_drop if col in df_final.columns]
if existing_cols_to_drop:
    print(f"\n[診斷] 偵測到並準備移除以下 {len(existing_cols_to_drop)} 個欄位: \n{existing_cols_to_drop}")
    df_final.drop(columns=existing_cols_to_drop, inplace=True)
    print(f"\n[成功] 已移除指定欄位。")
else:
    print("\n[資訊] 在資料中未找到任何需要移除的指定欄位。")

# ==============================================================================
# Part 5: Fill Missing Benchmark Values (Before Output)
# ==============================================================================
print("\n開始補空值...")

def _get_tax_rate(row):
    """Get the tax rate for actual carbon tax rounds."""
    if row['is_tax'] != 1:
        return 0.0
    if row['tax_rate_low'] == 1:
        return 3.0
    if row['tax_rate_high'] == 1:
        return 9.0
    if pd.notna(row.get('group_tax_rate')):
        return float(row['group_tax_rate'])
    return 0.0


def _get_policy_tax_rate(row):
    """Use the policy's effective carbon price: tax rate for tax, permit price for trade."""
    if row['is_tax'] == 1:
        return _get_tax_rate(row)
    if row['is_trade'] == 1:
        carbon_multiplier = row.get('group_carbon_multiplier')
        if pd.notna(carbon_multiplier) and float(carbon_multiplier) > 0:
            return float(carbon_multiplier) * 6.0
        if row.get('eq_price_low', 0) == 1:
            return 3.0
        if row.get('eq_price_high', 0) == 1:
            return 9.0
        return 6.0
    return 0.0


def _calculate_quantity_profit_emission(price, a, emission_per_unit, disturbances, max_q, incentive_rate, payment_rate=0.0):
    """Calculate best-response quantity plus private profit and emissions."""
    q_star = 0
    unit_incentive = emission_per_unit * incentive_rate
    unit_payment = emission_per_unit * payment_rate

    for idx in range(max_q):
        q = idx + 1
        if idx < len(disturbances):
            marginal_cost = a * q + disturbances[idx]
            if price > marginal_cost + unit_incentive:
                q_star = q

    if q_star == 0:
        total_cost = 0.0
    else:
        total_cost = 0.0
        for i in range(int(q_star)):
            if i < len(disturbances):
                total_cost += a * (i + 1) + disturbances[i]

    revenue = price * q_star
    policy_payment = unit_payment * q_star
    profit = int(round(revenue - total_cost - policy_payment))
    emissions = int(round(emission_per_unit * q_star))

    return int(q_star), profit, emissions


def calculate_player_benchmarks(row):
    """根據原始邏輯計算 player benchmarks"""
    try:
        # 解析 disturbance_values
        disturbance_values = eval(row['disturbance_values']) if pd.notna(row['disturbance_values']) else []
        disturbances = np.array(disturbance_values, dtype=float)
        
        max_q = int(row.get('max_production', len(disturbances) or 0))
        if len(disturbances) > 0:
            max_q = min(max_q, len(disturbances))
        else:
            max_q = 0
        
        price = float(row['market_price']) if pd.notna(row['market_price']) else 0
        a = float(row.get('marginal_cost_coefficient', 0) or 0)
        emission_per_unit = float(row.get('carbon_emission_per_unit', 0) or 0)
        social_cost_per_unit_carbon = 6.0
        tax_rate = _get_tax_rate(row)
        policy_tax_rate = _get_policy_tax_rate(row)

        q_mkt, profit_mkt, emissions_mkt = _calculate_quantity_profit_emission(
            price, a, emission_per_unit, disturbances, max_q, incentive_rate=0.0, payment_rate=0.0
        )
        q_soc, profit_soc, emissions_soc = _calculate_quantity_profit_emission(
            price, a, emission_per_unit, disturbances, max_q,
            incentive_rate=social_cost_per_unit_carbon, payment_rate=0.0
        )
        q_tax, profit_tax, emissions_tax = _calculate_quantity_profit_emission(
            price, a, emission_per_unit, disturbances, max_q,
            incentive_rate=tax_rate, payment_rate=tax_rate
        )
        q_policy, profit_policy, emissions_policy = _calculate_quantity_profit_emission(
            price, a, emission_per_unit, disturbances, max_q,
            incentive_rate=policy_tax_rate, payment_rate=policy_tax_rate
        )
        
        return {
            'q_soc': int(q_soc), 'q_mkt': int(q_mkt), 'q_tax': int(q_tax),
            'q_policy': int(q_policy),
            'pi_soc': int(profit_soc), 'pi_mkt': int(profit_mkt), 'pi_tax': int(profit_tax),
            'pi_policy': int(profit_policy),
            'e_soc': emissions_soc, 'e_mkt': emissions_mkt, 'e_tax': emissions_tax,
            'e_policy': emissions_policy,
        }
    except Exception as e:
        print(f"計算失敗 row {row.name}: {e}")
        return {
            'q_soc': 0, 'q_mkt': 0, 'q_tax': 0, 'q_policy': 0,
            'pi_soc': 0, 'pi_mkt': 0, 'pi_tax': 0, 'pi_policy': 0,
            'e_soc': 0, 'e_mkt': 0, 'e_tax': 0, 'e_policy': 0,
        }

# 檢查需要補值的欄位
player_benchmark_cols = ['e_mkt', 'e_soc', 'e_tax', 'pi_mkt', 'pi_soc', 'pi_tax', 'q_mkt', 'q_soc', 'q_tax']
group_benchmark_cols = ['group_E_mkt', 'group_E_soc', 'group_E_tax', 'group_Pi_mkt', 'group_Pi_soc', 'group_Pi_tax', 'group_Q_mkt', 'group_Q_soc', 'group_Q_tax']

existing_player_cols = [col for col in player_benchmark_cols if col in df_final.columns]
missing_player_cols = [col for col in existing_player_cols if df_final[col].isna().any()]

if missing_player_cols:
    print(f"發現 {len(missing_player_cols)} 個欄位有空值: {missing_player_cols}")
    filled_count = 0
    for idx, row in df_final.iterrows():
        needs_fill = any(pd.isna(row[col]) for col in missing_player_cols)
        if needs_fill:
            benchmarks = calculate_player_benchmarks(row)
            for col in missing_player_cols:
                if pd.isna(row[col]):
                    df_final.at[idx, col] = benchmarks[col]
            filled_count += 1
    print(f"已補完 {filled_count} 行的空值")

# 補 group-level 空值
existing_group_cols = [col for col in group_benchmark_cols if col in df_final.columns]
missing_group_cols = [col for col in existing_group_cols if df_final[col].isna().any()]

if missing_group_cols:
    print(f"補 group-level 空值: {missing_group_cols}")
    for group_name, group_data in df_final.groupby(['session_code', 'app', 'round']):
        for col in missing_group_cols:
            if col.startswith('group_E_'):
                player_col = 'e_' + col[8:].lower()  
            elif col.startswith('group_Pi_'):
                player_col = 'pi_' + col[9:].lower()  
            elif col.startswith('group_Q_'):
                player_col = 'q_' + col[8:].lower()   
            else:
                continue
                
            if player_col in df_final.columns:
                group_sum = group_data[player_col].sum()
                df_final.loc[group_data.index, col] = group_sum

print("空值補完！")

# Carbon-trade policy benchmark: use the trade round's theoretical permit price
# (carbon_multiplier * 6) as the equivalent tax rate.
policy_cols = ['q_policy', 'e_policy', 'pi_policy']
policy_benchmarks = df_final.apply(calculate_player_benchmarks, axis=1, result_type='expand')
for col in policy_cols:
    df_final[col] = pd.to_numeric(policy_benchmarks[col], errors='coerce').astype(float)

if all(col in df_final.columns for col in ['q_tax', 'e_tax', 'pi_tax']):
    tax_mask = df_final['is_tax'] == 1
    for policy_col, source_col in zip(policy_cols, ['q_tax', 'e_tax', 'pi_tax']):
        df_final.loc[tax_mask, policy_col] = pd.to_numeric(
            df_final.loc[tax_mask, source_col], errors='coerce'
        )

if all(col in df_final.columns for col in ['q_mkt', 'e_mkt', 'pi_mkt']):
    control_mask = (df_final['is_tax'] != 1) & (df_final['is_trade'] != 1)
    for policy_col, source_col in zip(policy_cols, ['q_mkt', 'e_mkt', 'pi_mkt']):
        df_final.loc[control_mask, policy_col] = pd.to_numeric(
            df_final.loc[control_mask, source_col], errors='coerce'
        )

policy_group_map = {
    'q_policy': 'group_Q_policy',
    'e_policy': 'group_E_policy',
    'pi_policy': 'group_Pi_policy',
}
for source_col, group_col in policy_group_map.items():
    df_final[group_col] = df_final.groupby(['session_code', 'app', 'round'])[source_col].transform('sum')

print("[成功] 已計算 policy benchmark: q_policy, e_policy, pi_policy")

# ==============================================================================
# Part 6: Finalize and Save to CSV
# ==============================================================================
new_dummies = ['is_tax', 'is_trade', 'is_equal', 'tax_rate_low', 'tax_rate_high', 'eq_price_low', 'eq_price_high']
id_cols = ['participant_code', 'session_code', 'subject_id', 'app', 'round']
other_cols = [col for col in df_final.columns if col not in id_cols and col not in new_dummies]
final_order = id_cols + new_dummies + sorted(other_cols)
df_final = df_final[final_order]

# 依照指定順序排序
df_final = df_final.sort_values(['session_code', 'subject_id', 'app', 'round']).reset_index(drop=True)

# ==============================================================================
# Part 6.5: Create surplus variables (在所有欄位都準備好之後)
# ==============================================================================
print("\n開始計算 surplus 變數...")

# Subject-level surplus: net_profit - (emission * 6)，但需考慮不同政策情境
if 'net_profit' in df_final.columns and 'emission' in df_final.columns:
    # pandas>=3: pivoted columns are object dtype, so cast the operands to float before arithmetic
    _net_profit = pd.to_numeric(df_final['net_profit'], errors='coerce')
    _emission = pd.to_numeric(df_final['emission'], errors='coerce')
    _carbon_tax_paid = (
        pd.to_numeric(df_final['carbon_tax_paid'], errors='coerce').fillna(0)
        if 'carbon_tax_paid' in df_final.columns else None
    )
    # 初始化 surplus 欄位
    df_final['surplus'] = 0.0
    
    # Control: surplus = net_profit - (emission * 6)
    control_mask = df_final['app'] == 'Stage_Control'
    df_final.loc[control_mask, 'surplus'] = (
        _net_profit[control_mask] - (_emission[control_mask] * 6)
    )
    
    # Tax: surplus = net_profit - (emission * 6) + carbon_tax_paid
    # 因為 net_profit 已扣除 carbon_tax_paid，但稅是轉移支付非社會成本，故加回
    tax_mask = df_final['app'] == 'Stage_CarbonTax'
    if _carbon_tax_paid is not None:
        df_final.loc[tax_mask, 'surplus'] = (
            _net_profit[tax_mask] - (_emission[tax_mask] * 6) + _carbon_tax_paid[tax_mask]
        )
    else:
        df_final.loc[tax_mask, 'surplus'] = (
            _net_profit[tax_mask] - (_emission[tax_mask] * 6)
        )
        print("[警告] 找不到 carbon_tax_paid 欄位，Tax 情境的 surplus 計算可能不正確")
    
    # Trading: surplus = net_profit - (emission * 6)
    # net_profit 已包含交易損益，交易是轉移支付不影響總剩餘
    trading_mask = df_final['app'] == 'Stage_CarbonTrading'
    df_final.loc[trading_mask, 'surplus'] = (
        _net_profit[trading_mask] - (_emission[trading_mask] * 6)
    )
    
    print(f"[成功] 已計算 subject-level 的 surplus 變數")
    print("  - Control: surplus = net_profit - (emission * 6)")
    print("  - Tax: surplus = net_profit - (emission * 6) + carbon_tax_paid")
    print("  - Trading: surplus = net_profit - (emission * 6)")
else:
    print(f"[警告] 找不到 net_profit 或 emission 欄位，無法計算 surplus")
    df_final['surplus'] = 0

# Subject-level surplus_soc: 社會最適產量下的剩餘
if 'pi_soc' in df_final.columns and 'e_soc' in df_final.columns:
    df_final['surplus_soc'] = df_final['pi_soc'] - (df_final['e_soc'] * 6)
    print("[成功] 已計算 subject-level 的 surplus_soc 變數")
    print("  - surplus_soc = pi_soc - (e_soc * 6)")
else:
    print("[警告] 找不到 pi_soc 或 e_soc 欄位，無法計算 surplus_soc")
    df_final['surplus_soc'] = 0

# Group-level surplus: sum of surplus for each group
if 'surplus' in df_final.columns:
    group_surplus = df_final.groupby(['session_code', 'app', 'round'])['surplus'].sum().reset_index()
    group_surplus.rename(columns={'surplus': 'group_surplus'}, inplace=True)
    df_final = pd.merge(df_final, group_surplus, on=['session_code', 'app', 'round'], how='left')
    print("[成功] 已計算 group-level 的 group_surplus 變數")
else:
    df_final['group_surplus'] = 0
    print("[警告] 無法計算 group_surplus")

# Group-level surplus_soc: sum of surplus_soc for each group
if 'surplus_soc' in df_final.columns:
    group_surplus_soc = df_final.groupby(['session_code', 'app', 'round'])['surplus_soc'].sum().reset_index()
    group_surplus_soc.rename(columns={'surplus_soc': 'group_surplus_soc'}, inplace=True)
    df_final = pd.merge(df_final, group_surplus_soc, on=['session_code', 'app', 'round'], how='left')
    print("[成功] 已計算 group-level 的 group_surplus_soc 變數")
else:
    df_final['group_surplus_soc'] = 0
    print("[警告] 無法計算 group_surplus_soc")

print("surplus 變數計算完成！")

save_errors = []

# 輸出 subj_lv_panel.csv 到 build/output
try:
    df_final.to_csv(output_file_path, index=False, encoding='utf-8-sig')
    print(f"[OK] 已輸出 subj_lv_panel.csv 至: {output_file_path}")
except Exception as e:
    save_errors.append(f"subj_lv_panel(build/output): {e}")

# 輸出 subject_lv_panel.csv 到 analysis/input
subj_lv_panel_path = os.path.join(SCRIPT_DIR, '..', '..', 'analysis', 'input', 'subj_lv_panel.csv')
try:
    df_final.to_csv(subj_lv_panel_path, index=False, encoding='utf-8-sig')
    print(f"[OK] 已額外輸出 subject_lv_panel.csv 至: {subj_lv_panel_path}")
except Exception as e:
    save_errors.append(f"subj_lv_panel(analysis/input): {e}")

# 產生 group_lv_panel.csv
group_keep = [
    'session_code', 'app', 'round',
    'is_tax', 'is_trade', 'is_equal',
    'tax_rate_low', 'tax_rate_high', 'eq_price_low', 'eq_price_high'
]
group_keep += [col for col in df_final.columns if col.startswith('group_')]
group_keep = list(dict.fromkeys(group_keep))  # 去重

df_group = df_final[group_keep].copy()
# 將 group_ 開頭的欄位名稱去除 group_ 前綴
new_columns = []
for col in df_group.columns:
    if col.startswith('group_'):
        new_columns.append(col[len('group_'):])
    else:
        new_columns.append(col)
df_group.columns = new_columns

# 計算每 group 的 net_profit 與 production 總和
sum_df = df_final.groupby(['session_code', 'app', 'round'], as_index=False)[['net_profit', 'production']].sum()

# 合併進 df_group
df_group = pd.merge(df_group, sum_df, on=['session_code', 'app', 'round'], how='left')

# 輸出 group_lv_panel.csv 到 output
group_lv_panel_path = os.path.join(OUTPUT_DIR, 'group_lv_panel.csv')
try:
    df_group.to_csv(group_lv_panel_path, index=False, encoding='utf-8-sig')
    print(f"[OK] 已輸出 group_lv_panel.csv 至: {group_lv_panel_path}")
except Exception as e:
    save_errors.append(f"group_lv_panel(build/output): {e}")

# 輸出 group_lv_panel.csv 到 analysis/input
group_lv_panel_path2 = os.path.join(SCRIPT_DIR, '..', '..', 'analysis', 'input', 'group_lv_panel.csv')
try:
    df_group.to_csv(group_lv_panel_path2, index=False, encoding='utf-8-sig')
    print(f"[OK] 已額外輸出 group_lv_panel.csv 至: {group_lv_panel_path2}")
except Exception as e:
    save_errors.append(f"group_lv_panel(analysis/input): {e}")

print(f"\n資料處理完成！")
print(f"最終的 Panel Data 共有 {df_final.shape[0]} 行, {df_final.shape[1]} 個欄位。")
if save_errors:
    print("[警告] 以下輸出失敗：")
    for err in save_errors:
        print(f"  - {err}")
else:
    print(f"結果已成功儲存至: {output_file_path}")
