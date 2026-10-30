import pandas as pd
import numpy as np
import os
import glob
import json # Library to parse the trade data string
import re

# ==============================================================================
# Part 0: Setup File Paths and Directories
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'input')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'output')
OUTPUT_FILENAME = 'trades_panel.csv'
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
# Part 2: Extract and Process Trade Data
# ==============================================================================
print("\n開始篩選並解析交易資料...")

all_trades_data = []
session_groups = df_wide.groupby('session.code')

for session_code, session_df in session_groups:
    print(f"  處理中 Session: {session_code}")
    
    allocation_method = session_df['session.config.allocation_method'].iloc[0]
    is_equal = 1 if allocation_method == 'equal' else 0
    
    for round_num in range(1, 13):
        trades_col = f'Stage_CarbonTrading.{round_num}.subsession.executed_trades'
        multiplier_col = f'Stage_CarbonTrading.{round_num}.subsession.carbon_multiplier'

        if trades_col not in session_df.columns or multiplier_col not in session_df.columns:
            continue

        trades_series = session_df[trades_col].dropna()
        if trades_series.empty:
            continue
        trades_string = trades_series.iloc[0]
        
        multiplier_series = session_df[multiplier_col].dropna()
        if multiplier_series.empty:
            continue
        carbon_multiplier = multiplier_series.iloc[0]

        if not isinstance(trades_string, str) or trades_string == '[]':
            continue

        # --- Calculate round-level variables ---
        original_multiplier = carbon_multiplier
        
        if pd.notna(original_multiplier):
            eq_price = original_multiplier * 6
        else:
            eq_price = None # Keep it as None if original is missing
        
        # Dummies are still based on the original multiplier values
        eq_price_low = 1 if original_multiplier == 0.5 else 0
        eq_price_high = 1 if original_multiplier == 1.5 else 0

        try:
            trades_list = json.loads(trades_string)
        except json.JSONDecodeError:
            print(f"    [警告] Session {session_code}, Round {round_num} 的交易資料格式錯誤，已跳過。")
            continue

        for trade_index, trade_dict in enumerate(trades_list, start=1):
            try:
                minutes, secs = map(int, trade_dict['timestamp'].split(':'))
                seconds = minutes * 60 + secs
            except (ValueError, AttributeError):
                seconds = None

            buyer_id = trade_dict.get('buyer_id')
            seller_id = trade_dict.get('seller_id')

            trade_row = {
                'session_code': session_code, 'is_equal': is_equal, 'eq_price': eq_price,
                'eq_price_low': eq_price_low, 'eq_price_high': eq_price_high, 'round': round_num,
                'trade_index': trade_index, 'seconds': seconds, 'buyer_id': buyer_id,
                'seller_id': seller_id, 'price': trade_dict.get('price'), 'quantity': trade_dict.get('quantity'),
                'buyer_is_dominant': 1 if buyer_id is not None and 1 <= buyer_id <= 3 else 0,
                'seller_is_dominant': 1 if seller_id is not None and 1 <= seller_id <= 3 else 0
            }
            all_trades_data.append(trade_row)

# ==============================================================================
# Part 3: Create Final DataFrame and Save
# ==============================================================================
if not all_trades_data:
    print("\n處理完成，但未在資料中找到任何有效的交易紀錄。")
    exit()

final_df = pd.DataFrame(all_trades_data)

final_column_order = [
    'session_code', 'round', 'trade_index', 'seconds', 'buyer_id', 'seller_id', 
    'price', 'quantity', 'is_equal', 'buyer_is_dominant', 'seller_is_dominant', 
    'eq_price', 'eq_price_low', 'eq_price_high'
]
final_df = final_df[final_column_order]

try:
    final_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
    print(f"\n資料處理完成！")
    print(f"最終的交易資料共有 {final_df.shape[0]} 行 (筆交易)。")
    print(f"結果已成功儲存至: {output_file_path}")

    # 輸出 trades_panel.csv 到 analysis/input
    analysis_input_path = os.path.join(SCRIPT_DIR, '..', '..', 'analysis', 'input', 'trades_panel.csv')
    final_df.to_csv(analysis_input_path, index=False, encoding='utf-8-sig')
    print(f"[OK] 已額外輸出 trades_panel.csv 至: {analysis_input_path}")
except Exception as e:
    print(f"\n錯誤：儲存檔案失敗。原因: {e}")

# ==============================================================================
# --- NEW: allocation/Hahn framing --- Part 4: Offers panel (every submitted order)
# One row per order a player submitted in a Carbon Trading round, from
# Stage_CarbonTrading.{r}.player.submitted_offers (JSON list of
# {"timestamp": "mm:ss", "direction": "buy"/"sell", "price", "quantity"}).
# Used by the friction diagnostics in analysis/code/run_trade_mkt_reg.py.
# ==============================================================================
OFFERS_FILENAME = 'offers_panel.csv'
all_offers = []
for session_code, session_df in session_groups:
    allocation_method = session_df['session.config.allocation_method'].iloc[0]
    is_equal = 1 if allocation_method == 'equal' else 0
    for round_num in range(1, 13):
        offers_col = f'Stage_CarbonTrading.{round_num}.player.submitted_offers'
        id_col = f'Stage_CarbonTrading.{round_num}.player.id_in_group'
        dom_col = f'Stage_CarbonTrading.{round_num}.player.is_dominant'
        multiplier_col = f'Stage_CarbonTrading.{round_num}.subsession.carbon_multiplier'
        if offers_col not in session_df.columns or multiplier_col not in session_df.columns:
            continue
        multiplier_series = session_df[multiplier_col].dropna()
        if multiplier_series.empty:
            continue
        carbon_multiplier = multiplier_series.iloc[0]
        eq_price = carbon_multiplier * 6 if pd.notna(carbon_multiplier) else None
        eq_price_low = 1 if carbon_multiplier == 0.5 else 0
        eq_price_high = 1 if carbon_multiplier == 1.5 else 0
        for _, prow in session_df.iterrows():
            raw_offers = prow.get(offers_col)
            if not isinstance(raw_offers, str) or raw_offers in ('', '[]'):
                continue
            try:
                offers_list = json.loads(raw_offers)
            except json.JSONDecodeError:
                print(f"    [警告] Session {session_code}, Round {round_num} 的掛單資料格式錯誤，已跳過。")
                continue
            pid = prow.get(id_col)
            is_dom = prow.get(dom_col)
            for offer_index, offer in enumerate(offers_list, start=1):
                try:
                    minutes, secs = map(int, str(offer.get('timestamp')).split(':'))
                    seconds = minutes * 60 + secs
                except (ValueError, AttributeError):
                    seconds = None
                all_offers.append({
                    'session_code': session_code, 'round': round_num, 'id_in_group': pid,
                    'is_dominant': 1 if (pd.notna(is_dom) and bool(is_dom)) else (1 if pd.notna(pid) and 1 <= pid <= 3 else 0),
                    'offer_index': offer_index, 'seconds': seconds,
                    'direction': offer.get('direction'), 'price': offer.get('price'), 'quantity': offer.get('quantity'),
                    'is_equal': is_equal, 'eq_price': eq_price, 'eq_price_low': eq_price_low, 'eq_price_high': eq_price_high,
                })

if all_offers:
    offers_df = pd.DataFrame(all_offers)[[
        'session_code', 'round', 'id_in_group', 'is_dominant', 'offer_index', 'seconds',
        'direction', 'price', 'quantity', 'is_equal', 'eq_price', 'eq_price_low', 'eq_price_high']]
    try:
        offers_out = os.path.join(OUTPUT_DIR, OFFERS_FILENAME)
        offers_df.to_csv(offers_out, index=False, encoding='utf-8-sig')
        offers_df.to_csv(os.path.join(SCRIPT_DIR, '..', '..', 'analysis', 'input', OFFERS_FILENAME), index=False, encoding='utf-8-sig')
        print(f"[OK] 已輸出 {OFFERS_FILENAME}（{offers_df.shape[0]} 筆掛單）至 build/output 與 analysis/input")
    except Exception as e:
        print(f"[警告] 無法輸出 {OFFERS_FILENAME}: {e}")
else:
    print("[資訊] 未找到任何掛單資料，未輸出 offers_panel.csv")
# --- END NEW ---
