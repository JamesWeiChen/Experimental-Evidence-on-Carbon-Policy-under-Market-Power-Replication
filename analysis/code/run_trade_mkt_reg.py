# %% 分析 MUDA 市場交易是否受 is_equal 影響
import os
import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.dirname(__file__))
from reg_utils import reg_func

# --- NEW: allocation/Hahn framing ---
RUN_NEW_ANALYSES = True   # set False to skip the additional analyses written to analysis/output/new/
# --- END NEW ---

# 參數與路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(BASE_DIR, "..", "input", "trades_panel.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
INITIAL_TRADE_COUNT = 3
LOG_GAP_EPSILON = 0.5
CONVERGENCE_REMAINING_SHARE = 0.75

# 讀資料與衍生欄
if not os.path.exists(INPUT):
	raise FileNotFoundError("trades_panel.csv not found in input directory.")
df = pd.read_csv(INPUT)

# 篩選只使用 round 1-12 的數據
df = df[(df["round"] >= 1) & (df["round"] <= 12)].copy()
print(f"[INFO] 篩選 round 1-12 後的資料筆數: {len(df)}")

df["price_diff"] = df["price"] - df["eq_price"]
df_initial_price_source = df.copy()

# 使用 IQR 方法在每個 session-round 內排除價格異常值
print("\n排除價格異常值 (使用 IQR 方法，按 session-round 分別處理)...")
df_before = len(df)

# 對每個 session-round 組合分別計算 IQR 並標記異常值
df["is_outlier"] = False

for (session, round_num), group in df.groupby(["session_code", "round"]):
	if len(group) < 4:  # 如果樣本數太少，不排除異常值
		continue
	
	Q1 = group["price_diff"].quantile(0.25)
	Q3 = group["price_diff"].quantile(0.75)
	IQR = Q3 - Q1
	lower_bound = Q1 - 1.5 * IQR
	upper_bound = Q3 + 1.5 * IQR
	
	# 標記該回合中的異常值
	outlier_mask = (group["price_diff"] < lower_bound) | (group["price_diff"] > upper_bound)
	df.loc[group.index[outlier_mask], "is_outlier"] = True

# 移除所有被標記為異常值的數據
outliers_removed = df["is_outlier"].sum()
df = df[~df["is_outlier"]].copy()
df = df.drop(columns=["is_outlier"])
df_after = len(df)

print(f"  移除 {outliers_removed} 筆異常值 ({outliers_removed/df_before*100:.2f}%)")
print(f"  保留 {df_after} 筆數據 ({df_after/df_before*100:.2f}%)")
print("  (異常值判定標準：在每個 session-round 內使用 Q1-1.5×IQR 和 Q3+1.5×IQR)\n")

df["mins"] = df["seconds"] / 60
df["cluster_round"] = df["session_code"].astype(str) + "_" + df["round"].astype(str)
df["abs_price_diff"] = df["price_diff"].abs()
df["log_abs_price_diff"] = np.log1p(df["abs_price_diff"])


def _zscore(series):
	std = series.std(ddof=0)
	if pd.isna(std) or std == 0:
		return pd.Series(0.0, index=series.index)
	return (series - series.mean()) / std


def make_initial_convergence_panel(
	data,
	initial_source,
	initial_trade_count=INITIAL_TRADE_COUNT,
	log_gap_epsilon=LOG_GAP_EPSILON,
):
	"""Build post-initial trade panel using raw first trades as initial signals."""
	outcome_required_cols = {
		"session_code",
		"round",
		"trade_index",
		"seconds",
		"price",
		"price_diff",
		"abs_price_diff",
		"log_abs_price_diff",
	}
	initial_required_cols = {
		"session_code",
		"round",
		"trade_index",
		"seconds",
		"price",
		"price_diff",
	}
	missing_cols = sorted(outcome_required_cols - set(data.columns))
	if missing_cols:
		raise ValueError(
			"Post-initial trade data is missing required columns for initial convergence analysis: "
			+ ", ".join(missing_cols)
		)
	missing_cols = sorted(initial_required_cols - set(initial_source.columns))
	if missing_cols:
		raise ValueError(
			"Initial trade source data is missing required columns for initial convergence analysis: "
			+ ", ".join(missing_cols)
		)

	group_cols = ["session_code", "round"]
	initial_ordered = initial_source.sort_values(
		group_cols + ["trade_index", "seconds"]
	).copy()
	initial_trades = initial_ordered.groupby(group_cols, group_keys=False).head(
		initial_trade_count
	)
	initial = (
		initial_trades.groupby(group_cols)
		.agg(
			initial_trade_count_observed=("trade_index", "size"),
			initial_last_trade_index=("trade_index", "max"),
			initial_cutoff_seconds=("seconds", "max"),
			initial_avg_price=("price", "mean"),
			initial_price_diff=("price_diff", "mean"),
			initial_abs_price_diff=("price_diff", lambda s: s.abs().mean()),
		)
		.reset_index()
	)
	initial = initial[
		(initial["initial_trade_count_observed"] == initial_trade_count)
		& (initial["initial_abs_price_diff"] > 0)
	].copy()
	initial["initial_price_diff_z"] = _zscore(initial["initial_price_diff"])
	initial["initial_abs_price_diff_z"] = _zscore(initial["initial_abs_price_diff"])

	ordered = data.sort_values(group_cols + ["trade_index", "seconds"]).copy()
	panel = ordered.merge(initial, on=group_cols, how="inner")
	panel = panel[panel["trade_index"] > panel["initial_last_trade_index"]].copy()
	panel["post_initial_trade_index"] = (
		panel["trade_index"] - panel["initial_last_trade_index"]
	)
	panel["post_initial_seconds"] = (
		panel["seconds"] - panel["initial_cutoff_seconds"]
	).clip(lower=0)
	panel["post_initial_mins"] = panel["post_initial_seconds"] / 60
	panel["remaining_gap_pct"] = (
		panel["abs_price_diff"] / panel["initial_abs_price_diff"]
	)
	panel["gap_closed_pct"] = 1 - panel["remaining_gap_pct"]
	panel["log_remaining_gap_ratio"] = np.log(
		(panel["abs_price_diff"] + log_gap_epsilon)
		/ (panel["initial_abs_price_diff"] + log_gap_epsilon)
	)

	analysis_cols = [
		"abs_price_diff",
		"log_abs_price_diff",
		"remaining_gap_pct",
		"gap_closed_pct",
		"log_remaining_gap_ratio",
		"price_diff",
		"post_initial_mins",
		"initial_abs_price_diff_z",
		"initial_price_diff_z",
		"is_equal",
		"eq_price_low",
		"eq_price_high",
		"round",
		"session_code",
	]
	return panel.dropna(subset=analysis_cols).copy()


def make_convergence_time_panel(
	panel,
	remaining_share=CONVERGENCE_REMAINING_SHARE,
):
	"""Session-round level time to reach a fixed remaining share of the initial gap."""
	group_cols = ["session_code", "round"]
	threshold_name = "convergence_threshold"
	rows = []
	for (session_code, round_num), group in panel.groupby(group_cols):
		group = group.sort_values(["post_initial_seconds", "trade_index"]).copy()
		reached = group[group["remaining_gap_pct"] <= remaining_share]
		first = group.iloc[0]
		row = {
			"session_code": session_code,
			"round": round_num,
			"is_equal": first["is_equal"],
			"eq_price_low": first["eq_price_low"],
			"eq_price_high": first["eq_price_high"],
			"initial_abs_price_diff": first["initial_abs_price_diff"],
			"initial_abs_price_diff_z": first["initial_abs_price_diff_z"],
			"initial_price_diff": first["initial_price_diff"],
			"initial_price_diff_z": first["initial_price_diff_z"],
			"post_initial_trade_count": len(group),
			"last_observed_post_initial_mins": group["post_initial_mins"].max(),
			"convergence_remaining_share": remaining_share,
			"convergence_closed_share": 1 - remaining_share,
		}
		if reached.empty:
			row[f"reached_{threshold_name}"] = 0
			row[f"time_to_{threshold_name}_mins"] = np.nan
			row[f"time_to_{threshold_name}_mins_censored"] = row[
				"last_observed_post_initial_mins"
			]
		else:
			row[f"reached_{threshold_name}"] = 1
			row[f"time_to_{threshold_name}_mins"] = reached.iloc[0][
				"post_initial_mins"
			]
			row[f"time_to_{threshold_name}_mins_censored"] = row[
				f"time_to_{threshold_name}_mins"
			]
		rows.append(row)

	return pd.DataFrame(rows)

# 回歸模型設計
formulas = [
    # baseline
    # "price_diff ~ is_equal + mins + round",
    # "price_diff ~ is_equal + mins + round + buyer_is_dominant + seller_is_dominant",
    # "price_diff ~ is_equal + mins + round + buyer_is_dominant + seller_is_dominant"
    # " + is_equal*buyer_is_dominant + is_equal*seller_is_dominant",
    # "price_diff ~ is_equal + mins + round + eq_price_low + eq_price_high",
    # "price_diff ~ is_equal + mins + round + eq_price_low + eq_price_high + is_equal*mins",

    # 二階交互作用模型
    "price_diff ~ is_equal * (eq_price_low + eq_price_high + mins) + round",

    # 三階交互作用模型
    "price_diff ~ is_equal * (eq_price_low + eq_price_high) * mins + round",
    
    # === 新增：檢驗 dominant firm 市場力量的模型 ===
    
    # Model 3: 基本 dominant firm 效果
    "price_diff ~ is_equal * (buyer_is_dominant + seller_is_dominant) + mins + round",
    
    # Model 4: Dominant firm 與均衡價格的交互作用
    "price_diff ~ is_equal * (buyer_is_dominant + seller_is_dominant) * (eq_price_low + eq_price_high) + mins + round",
    
    # Model 5: 檢驗 dominant firm 在不同時間點的影響（是否隨時間收斂）
    "price_diff ~ is_equal * (buyer_is_dominant + seller_is_dominant) * mins + round",
]


# 組成 y_to_formulas 結構
from collections import defaultdict
y_to_formulas = defaultdict(list)
for fml in formulas:
	y = fml.split("~")[0].strip()
	y_to_formulas[y].append(fml)

# Wald/simple effects callback for reg_func
def trade_simple_effects(res, idx, label):
	lines = []
	params = res.params.index.astype(str)
	
	# === 原有的 eq_price 交互作用檢定 ===
	if ("is_equal:eq_price_low" in params.values) and ("is_equal:eq_price_high" in params.values):
		# eq_price=6（基準）
		tt6 = res.t_test("is_equal = 0")
		lines.append(f"eq_price=6: effect={tt6.effect.item():.3f}, SE={tt6.sd.item():.3f}, t={tt6.tvalue.item():.3f}, p={tt6.pvalue.item():.3f}")
		# eq_price=3
		tt3 = res.t_test("is_equal + is_equal:eq_price_low = 0")
		lines.append(f"eq_price=3: effect={tt3.effect.item():.3f}, SE={tt3.sd.item():.3f}, t={tt3.tvalue.item():.3f}, p={tt3.pvalue.item():.3f}")
		# eq_price=9
		tt9 = res.t_test("is_equal + is_equal:eq_price_high = 0")
		lines.append(f"eq_price=9: effect={tt9.effect.item():.3f}, SE={tt9.sd.item():.3f}, t={tt9.tvalue.item():.3f}, p={tt9.pvalue.item():.3f}")
		# Wald 檢定：異質性
		wald = res.wald_test([
			"is_equal:eq_price_low = 0",
			"is_equal:eq_price_high = 0",
		])
		lines.append(f"Wald heterogeneity: chi2={float(np.asarray(wald.statistic).squeeze()):.3f}, df={int(wald.df_denom)}, p={wald.pvalue:.3f}")
	
	if ("is_equal:eq_price_low:mins" in params.values) and ("is_equal:eq_price_high:mins" in params.values):
		# Wald 檢定：異質性
		wald = res.wald_test([
			"is_equal:eq_price_low:mins = 0",
			"is_equal:eq_price_high:mins = 0",
		])
		lines.append(f"Wald heterogeneity (mins): chi2={float(np.asarray(wald.statistic).squeeze()):.3f}, df={int(wald.df_denom)}, p={wald.pvalue:.3f}")
	
	# === 新增：dominant firm 市場力量檢定 ===
	if ("is_equal:buyer_is_dominant" in params.values) and ("is_equal:seller_is_dominant" in params.values):
		lines.append("\n--- Dominant Firm Market Power Tests ---")
		
		# is_equal=0 時的效果
		if "is_equal" in params.values and "buyer_is_dominant" in params.values:
			tt_buyer_equal0 = res.t_test("buyer_is_dominant = 0")
			lines.append(f"Buyer dominant (is_equal=0): coef={tt_buyer_equal0.effect.item():.3f}, SE={tt_buyer_equal0.sd.item():.3f}, t={tt_buyer_equal0.tvalue.item():.3f}, p={tt_buyer_equal0.pvalue.item():.3f}")
		
		if "seller_is_dominant" in params.values:
			tt_seller_equal0 = res.t_test("seller_is_dominant = 0")
			lines.append(f"Seller dominant (is_equal=0): coef={tt_seller_equal0.effect.item():.3f}, SE={tt_seller_equal0.sd.item():.3f}, t={tt_seller_equal0.tvalue.item():.3f}, p={tt_seller_equal0.pvalue.item():.3f}")
		
		# is_equal=1 時的效果
		tt_buyer_equal1 = res.t_test("buyer_is_dominant + is_equal:buyer_is_dominant = 0")
		lines.append(f"Buyer dominant (is_equal=1): coef={tt_buyer_equal1.effect.item():.3f}, SE={tt_buyer_equal1.sd.item():.3f}, t={tt_buyer_equal1.tvalue.item():.3f}, p={tt_buyer_equal1.pvalue.item():.3f}")
		
		tt_seller_equal1 = res.t_test("seller_is_dominant + is_equal:seller_is_dominant = 0")
		lines.append(f"Seller dominant (is_equal=1): coef={tt_seller_equal1.effect.item():.3f}, SE={tt_seller_equal1.sd.item():.3f}, t={tt_seller_equal1.tvalue.item():.3f}, p={tt_seller_equal1.pvalue.item():.3f}")
		
		# Wald 檢定：is_equal=0 vs is_equal=1 的差異
		wald_buyer = res.wald_test("is_equal:buyer_is_dominant = 0")
		wald_seller = res.wald_test("is_equal:seller_is_dominant = 0")
		lines.append(f"\nWald test - Buyer dominant effect differs by is_equal: chi2={float(np.asarray(wald_buyer.statistic).squeeze()):.3f}, p={wald_buyer.pvalue:.3f}")
		lines.append(f"Wald test - Seller dominant effect differs by is_equal: chi2={float(np.asarray(wald_seller.statistic).squeeze()):.3f}, p={wald_seller.pvalue:.3f}")
		
		# 聯合檢定：兩個交互項是否同時為0
		wald_joint = res.wald_test([
			"is_equal:buyer_is_dominant = 0",
			"is_equal:seller_is_dominant = 0",
		])
		lines.append(f"Wald test - Joint test (both interactions): chi2={float(np.asarray(wald_joint.statistic).squeeze()):.3f}, df={int(wald_joint.df_denom)}, p={wald_joint.pvalue:.3f}")

	return lines if lines else None


def _find_interaction(params, left, right):
	param_values = set(params.values)
	for term in (f"{left}:{right}", f"{right}:{left}"):
		if term in param_values:
			return term
	return None


def _format_ttest(prefix, test_result):
	return (
		f"{prefix}: coef={test_result.effect.item():.3f}, "
		f"SE={test_result.sd.item():.3f}, "
		f"t={test_result.tvalue.item():.3f}, "
		f"p={test_result.pvalue.item():.3f}"
	)


def convergence_simple_effects(res, idx, label):
	lines = []
	params = res.params.index.astype(str)
	if "post_initial_mins" not in params.values:
		return None

	for initial_var, var_label in [
		("initial_abs_price_diff_z", "initial absolute price gap"),
		("initial_price_diff_z", "initial signed price gap"),
	]:
		interaction = _find_interaction(params, initial_var, "post_initial_mins")
		if interaction is None:
			continue

		lines.append(f"\n--- Initial Price Convergence Tests ({var_label}) ---")
		lines.append(
			_format_ttest(
				f"Time slope when {var_label} is average (z=0)",
				res.t_test("post_initial_mins = 0"),
			)
		)
		lines.append(
			_format_ttest(
				f"Time slope when {var_label} is high (z=+1)",
				res.t_test(f"post_initial_mins + {interaction} = 0"),
			)
		)
		lines.append(
			_format_ttest(
				f"Time slope when {var_label} is low (z=-1)",
				res.t_test(f"post_initial_mins - {interaction} = 0"),
			)
		)
		wald = res.wald_test(f"{interaction} = 0")
		lines.append(
			f"Wald test - initial price affects convergence speed: "
			f"chi2={float(np.asarray(wald.statistic).squeeze()):.3f}, p={wald.pvalue:.3f}"
		)

	return lines if lines else None

# ============================================================
# 分析 1: 完整數據（round 1-12）
# ============================================================
print(f"\n{'='*80}")
print("分析 1: 使用完整數據（round 1-12）")
print(f"{'='*80}")
reg_func(df, y_to_formulas, os.path.join(OUTPUT_DIR, "trade_mkt"), cluster_col="session_code", extra_effects_func=trade_simple_effects)

# ============================================================
# 分析 2: 只使用最後 60 秒的數據
# ============================================================
print(f"\n{'='*80}")
print("分析 2: 只使用最後 60 秒的交易數據")
print(f"{'='*80}")

# 篩選最後 60 秒的數據
df_last60 = df[df["seconds"] >= 60].copy()
print(f"[INFO] 最後 60 秒的資料筆數: {len(df_last60)}")

# 重新組成 y_to_formulas（相同的模型）
y_to_formulas_last60 = defaultdict(list)
for fml in formulas:
	y = fml.split("~")[0].strip()
	y_to_formulas_last60[y].append(fml)

reg_func(df_last60, y_to_formulas_last60, os.path.join(OUTPUT_DIR, "trade_mkt_last60"), cluster_col="session_code", extra_effects_func=trade_simple_effects)

# ============================================================
# 分析 3: 初始成交價格是否影響後續往理論價格收斂的速度
# ============================================================
print(f"\n{'='*80}")
print(f"分析 3: 前 {INITIAL_TRADE_COUNT} 筆成交價格與後續收斂速度")
print(f"{'='*80}")

df_initial_convergence = make_initial_convergence_panel(
	df,
	initial_source=df_initial_price_source,
	initial_trade_count=INITIAL_TRADE_COUNT,
)
eligible_rounds = df_initial_convergence[["session_code", "round"]].drop_duplicates().shape[0]
print(f"[INFO] 初始成交定義: 每個 session-round 前 {INITIAL_TRADE_COUNT} 筆 raw 成交（不排除 outlier）")
print(f"[INFO] 後續成交樣本數: {len(df_initial_convergence)}")
print(f"[INFO] 可用 session-round 數: {eligible_rounds}")
print(
	"[INFO] 只估計公平基準：相對於初始偏離的百分比收斂；"
	"不再用 raw price gap 斜率比較收斂速度。"
)

initial_convergence_formulas = [
	# gap_closed_pct = 1 - abs(price - eq_price) / initial_abs_price_diff.
	# 係數可解讀為每分鐘多收斂多少初始偏離比例。
	"gap_closed_pct ~ initial_abs_price_diff_z * post_initial_mins + is_equal + eq_price_low + eq_price_high + round",
	"gap_closed_pct ~ initial_abs_price_diff_z * post_initial_mins + is_equal * post_initial_mins + eq_price_low + eq_price_high + round",
	"gap_closed_pct ~ initial_abs_price_diff_z * post_initial_mins + is_equal * post_initial_mins + (eq_price_low + eq_price_high) * post_initial_mins + round",

	# log_remaining_gap_ratio = log((post_gap + eps) / (initial_gap + eps)).
	# 係數近似比例收斂率，eps 避免後續價差為 0 時無法取 log。
	"log_remaining_gap_ratio ~ initial_abs_price_diff_z * post_initial_mins + is_equal + eq_price_low + eq_price_high + round",
	"log_remaining_gap_ratio ~ initial_abs_price_diff_z * post_initial_mins + is_equal * post_initial_mins + eq_price_low + eq_price_high + round",
	"log_remaining_gap_ratio ~ initial_abs_price_diff_z * post_initial_mins + is_equal * post_initial_mins + (eq_price_low + eq_price_high) * post_initial_mins + round",
]

y_to_formulas_initial_convergence = defaultdict(list)
for fml in initial_convergence_formulas:
	y = fml.split("~")[0].strip()
	y_to_formulas_initial_convergence[y].append(fml)

reg_func(
	df_initial_convergence,
	y_to_formulas_initial_convergence,
	os.path.join(OUTPUT_DIR, "trade_mkt_initial_convergence"),
	cluster_col="session_code",
	extra_effects_func=convergence_simple_effects,
)

df_convergence_time = make_convergence_time_panel(df_initial_convergence)
closed_share_pct = int(round((1 - CONVERGENCE_REMAINING_SHARE) * 100))
remaining_share_pct = int(round(CONVERGENCE_REMAINING_SHARE * 100))
reached_threshold = int(
	df_convergence_time["reached_convergence_threshold"].sum()
)
time_panel_path = os.path.join(
	OUTPUT_DIR,
	"trade_mkt_initial_convergence_time_to_threshold_panel.csv",
)
df_convergence_time.to_csv(time_panel_path, index=False)
print(
	f"[INFO] 收斂至少 {closed_share_pct}% "
	f"（後續偏離 <= 初始偏離 {remaining_share_pct}%）的 session-round: "
	f"{reached_threshold}/{len(df_convergence_time)}"
)
print(f"[INFO] time-to-threshold panel 已輸出: {time_panel_path}")

time_to_convergence_outcomes = [
	"reached_convergence_threshold",
	"time_to_convergence_threshold_mins",
	"time_to_convergence_threshold_mins_censored",
]
time_to_convergence_rhs = [
	"initial_abs_price_diff_z + is_equal + eq_price_low + eq_price_high + round",
	"initial_abs_price_diff_z * is_equal + eq_price_low + eq_price_high + round",
	"initial_abs_price_diff_z * (is_equal + eq_price_low + eq_price_high) + round",
]
time_to_convergence_formulas = [
	f"{outcome} ~ {rhs}"
	for outcome in time_to_convergence_outcomes
	for rhs in time_to_convergence_rhs
]

y_to_formulas_time_to_convergence = defaultdict(list)
for fml in time_to_convergence_formulas:
	y = fml.split("~")[0].strip()
	y_to_formulas_time_to_convergence[y].append(fml)

time_to_convergence_y_to_df = {
	"reached_convergence_threshold": df_convergence_time.dropna(
		subset=["reached_convergence_threshold"]
	).copy(),
	"time_to_convergence_threshold_mins": df_convergence_time.dropna(
		subset=["time_to_convergence_threshold_mins"]
	).copy(),
	"time_to_convergence_threshold_mins_censored": df_convergence_time.dropna(
		subset=["time_to_convergence_threshold_mins_censored"]
	).copy(),
}

reg_func(
	df_convergence_time,
	y_to_formulas_time_to_convergence,
	os.path.join(OUTPUT_DIR, "trade_mkt_initial_convergence_time_to_threshold"),
	cluster_col="session_code",
	y_to_df=time_to_convergence_y_to_df,
)

print(f"\n{'='*80}")
print("所有分析完成！")
print(f"{'='*80}")


# ============================================================
# --- NEW: allocation/Hahn framing ---
# Signed price deviations, trader-pair prices, dispersion, and trading-friction diagnostics.
# Outputs go to analysis/output/new/ (no plots into doc/input, no changes to existing files).
# ============================================================
from reg_utils import new_output_dir, new_reg_table, write_text_header

EQ_DUMMIES = "eq_price_low + eq_price_high"
ALLOC_HYP = {  # simple effects of equal allocation at each theoretical price, and their heterogeneity
    "eq6": "is_equal = 0",
    "eq3": "is_equal + is_equal:eq_price_low = 0",
    "eq9": "is_equal + is_equal:eq_price_high = 0",
}


def _prep_trade_frame(d):
    d = d.copy()
    d["signed_diff"] = d["price"] - d["eq_price"]
    d["rel_dev"] = d["price"] / d["eq_price"] - 1.0
    d["mins"] = d["seconds"] / 60.0
    d["sec_bin30"] = (d["seconds"] // 30 * 30).clip(upper=90).astype(int)
    d["round_block"] = np.where(d["round"] <= 6, "rounds 1-6", "rounds 7-12")
    b = d["buyer_is_dominant"].astype(int) == 1
    s = d["seller_is_dominant"].astype(int) == 1
    d["pair_type"] = np.select([b & s, b & ~s, ~b & s], ["Db_Ds", "Db_Ss", "Sb_Ds"], default="Sb_Ss")
    for pt in ["Db_Ss", "Sb_Ds", "Db_Ds"]:
        d[pt] = (d["pair_type"] == pt).astype(int)
        d[pt + "_eq"] = d[pt] * d["is_equal"]
    d["cluster_round"] = d["session_code"].astype(str) + "_" + d["round"].astype(str)
    return d


def _alloc_by_price_hypotheses(prefix_terms=("is_equal",)):
    return [ALLOC_HYP["eq6"], ALLOC_HYP["eq3"], ALLOC_HYP["eq9"], "is_equal:eq_price_low = 0, is_equal:eq_price_high = 0"]


def _round_grid(trades_all, offers):
    keys = ["session_code", "round", "is_equal", "eq_price", "eq_price_low", "eq_price_high"]
    grid = trades_all[keys].drop_duplicates()
    if offers is not None and len(offers):
        grid = pd.concat([grid, offers[keys].drop_duplicates()], ignore_index=True).drop_duplicates(["session_code", "round"])
    return grid.sort_values(["session_code", "round"]).reset_index(drop=True)


def _round_level_prices(trades_all, grid):
    rows = []
    for (s, r), x in trades_all.sort_values(["session_code", "round", "trade_index"]).groupby(["session_code", "round"]):
        rows.append(dict(session_code=s, round=r, n_trades=len(x), qty_traded=int(x["quantity"].sum()),
                         open_diff=float(x["signed_diff"].iloc[0]), close_diff=float(x["signed_diff"].iloc[-1]),
                         vwap_diff=float(np.average(x["signed_diff"], weights=x["quantity"])), mean_diff=float(x["signed_diff"].mean()),
                         price_sd=float(x["price"].std(ddof=0)), price_iqr=float(x["price"].quantile(0.75) - x["price"].quantile(0.25)),
                         price_range=float(x["price"].max() - x["price"].min()), first_trade_sec=float(x["seconds"].min()),
                         share_last30=float((x["seconds"] >= 90).mean()), share_qty_ge2=float((x["quantity"] >= 2).mean())))
    out = grid.merge(pd.DataFrame(rows), on=["session_code", "round"], how="left")
    out["n_trades"] = out["n_trades"].fillna(0).astype(int)
    out["qty_traded"] = out["qty_traded"].fillna(0).astype(int)
    return out


def _dominant_dispersion(trades_all):
    rows = []
    for side, idcol in (("buyer", "buyer_id"), ("seller", "seller_id")):
        sub = trades_all[trades_all[idcol].between(1, 3)]
        for (s, r, pid), x in sub.groupby(["session_code", "round", idcol]):
            rows.append(dict(session_code=s, round=r, id_in_group=int(pid), side=side, is_equal=int(x["is_equal"].iloc[0]),
                             eq_price=float(x["eq_price"].iloc[0]), n_trades=len(x), qty=int(x["quantity"].sum()),
                             price_sd=float(x["price"].std(ddof=0)) if len(x) > 1 else np.nan,
                             price_range=float(x["price"].max() - x["price"].min()), mean_signed_diff=float(x["signed_diff"].mean())))
    return pd.DataFrame(rows)


def _replay_round(off, tr):
    """Best-effort order-book replay with the game's rules (price crosses AND quantity equal, no partial fills).
    Returns counts of crossing-but-quantity-mismatched orders and how many actual trades the replay reproduces."""
    off = off.sort_values(["seconds", "id_in_group", "offer_index"])
    bids, asks, replayed = [], [], []
    mism_bid = mism_ask = cross_bid = cross_ask = 0
    for o in off.itertuples(index=False):
        pid, price, qty, sec = o.id_in_group, o.price, o.quantity, o.seconds
        if o.direction == "buy":
            cands = [a for a in asks if a["price"] <= price and a["id"] != pid]
            if cands:
                cross_bid += 1
                same = [a for a in cands if a["qty"] == qty]
                if same:
                    a = min(same, key=lambda z: (z["price"], z["sec"]))
                    asks.remove(a)
                    replayed.append(dict(buyer=pid, seller=a["id"], price=a["price"], qty=qty, sec=sec))
                    continue
                mism_bid += 1
            bids.append(dict(id=pid, price=price, qty=qty, sec=sec))
        else:
            cands = [b for b in bids if b["price"] >= price and b["id"] != pid]
            if cands:
                cross_ask += 1
                same = [b for b in cands if b["qty"] == qty]
                if same:
                    b = max(same, key=lambda z: (z["price"], -z["sec"]))
                    bids.remove(b)
                    replayed.append(dict(buyer=b["id"], seller=pid, price=b["price"], qty=qty, sec=sec))
                    continue
                mism_ask += 1
            asks.append(dict(id=pid, price=price, qty=qty, sec=sec))
    matched = matched_loose = 0
    used = set()
    for t in tr.itertuples(index=False):
        for j, rp in enumerate(replayed):
            if j in used:
                continue
            if rp["buyer"] == t.buyer_id and rp["seller"] == t.seller_id and rp["qty"] == t.quantity and abs(rp["sec"] - t.seconds) <= 2:
                matched_loose += 1
                if rp["price"] == t.price:
                    matched += 1
                used.add(j)
                break
    return dict(n_offers=len(off), n_bids=int((off["direction"] == "buy").sum()), n_asks=int((off["direction"] == "sell").sum()),
                n_replayed_trades=len(replayed), n_actual_trades=len(tr), replay_matched=matched, replay_matched_loose=matched_loose,
                crossing_bids=cross_bid, crossing_bids_qty_mismatch=mism_bid, crossing_asks=cross_ask, crossing_asks_qty_mismatch=mism_ask)


def _friction_tables(offers, trades_all, grid):
    """Per player-round order and fill statistics (set difference: submitted orders minus executed trades)."""
    o = offers.copy()
    o["is_bid"] = (o["direction"] == "buy").astype(int)
    po = o.groupby(["session_code", "round", "id_in_group"]).agg(
        is_dominant=("is_dominant", "first"), n_bids=("is_bid", "sum"), n_asks=("is_bid", lambda s: int((s == 0).sum())),
        qty_bid=("quantity", lambda q: int(q[o.loc[q.index, "is_bid"] == 1].sum())),
        qty_ask=("quantity", lambda q: int(q[o.loc[q.index, "is_bid"] == 0].sum())),
        share_orders_qty_ge2=("quantity", lambda q: float((q >= 2).mean())), mean_order_qty=("quantity", "mean")).reset_index()
    tb = trades_all.groupby(["session_code", "round", "buyer_id"]).agg(n_buy_trades=("quantity", "size"), qty_bought=("quantity", "sum")).reset_index().rename(columns={"buyer_id": "id_in_group"})
    ts = trades_all.groupby(["session_code", "round", "seller_id"]).agg(n_sell_trades=("quantity", "size"), qty_sold=("quantity", "sum")).reset_index().rename(columns={"seller_id": "id_in_group"})
    pr = po.merge(tb, on=["session_code", "round", "id_in_group"], how="left").merge(ts, on=["session_code", "round", "id_in_group"], how="left")
    for c in ["n_buy_trades", "qty_bought", "n_sell_trades", "qty_sold"]:
        pr[c] = pr[c].fillna(0).astype(int)
    pr["unfilled_bids"] = (pr["n_bids"] - pr["n_buy_trades"]).clip(lower=0)
    pr["unfilled_asks"] = (pr["n_asks"] - pr["n_sell_trades"]).clip(lower=0)
    pr["fill_rate_bids"] = np.where(pr["n_bids"] > 0, pr["n_buy_trades"] / pr["n_bids"].replace(0, np.nan), np.nan)
    pr["fill_rate_asks"] = np.where(pr["n_asks"] > 0, pr["n_sell_trades"] / pr["n_asks"].replace(0, np.nan), np.nan)
    pr["fill_rate_bid_qty"] = np.where(pr["qty_bid"] > 0, pr["qty_bought"] / pr["qty_bid"].replace(0, np.nan), np.nan)
    pr["fill_rate_ask_qty"] = np.where(pr["qty_ask"] > 0, pr["qty_sold"] / pr["qty_ask"].replace(0, np.nan), np.nan)
    pr = pr.merge(grid, on=["session_code", "round"], how="left")
    # aggregate by firm type x allocation x eq_price (ratios of sums and means of player-round rates)
    agg = pr.groupby(["is_dominant", "is_equal", "eq_price"]).agg(
        player_rounds=("id_in_group", "size"), bids=("n_bids", "sum"), asks=("n_asks", "sum"),
        buy_trades=("n_buy_trades", "sum"), sell_trades=("n_sell_trades", "sum"),
        qty_bid=("qty_bid", "sum"), qty_bought=("qty_bought", "sum"), qty_ask=("qty_ask", "sum"), qty_sold=("qty_sold", "sum"),
        mean_fill_rate_bids=("fill_rate_bids", "mean"), mean_fill_rate_asks=("fill_rate_asks", "mean"),
        mean_share_orders_qty_ge2=("share_orders_qty_ge2", "mean"), mean_order_qty=("mean_order_qty", "mean")).reset_index()
    agg["fill_rate_bids_pooled"] = agg["buy_trades"] / agg["bids"].replace(0, np.nan)
    agg["fill_rate_asks_pooled"] = agg["sell_trades"] / agg["asks"].replace(0, np.nan)
    agg["fill_rate_bid_qty_pooled"] = agg["qty_bought"] / agg["qty_bid"].replace(0, np.nan)
    agg["fill_rate_ask_qty_pooled"] = agg["qty_sold"] / agg["qty_ask"].replace(0, np.nan)
    agg["unfilled_bids"] = agg["bids"] - agg["buy_trades"]
    agg["unfilled_asks"] = agg["asks"] - agg["sell_trades"]
    # order-book replay per session-round
    rep = []
    for (s, r), off in offers.groupby(["session_code", "round"]):
        tr = trades_all[(trades_all["session_code"] == s) & (trades_all["round"] == r)]
        d = _replay_round(off, tr)
        d.update(session_code=s, round=r)
        rep.append(d)
    rep = grid.merge(pd.DataFrame(rep), on=["session_code", "round"], how="left")
    rep["replay_match_share"] = rep["replay_matched"] / rep["n_actual_trades"].replace(0, np.nan)
    rep["replay_match_share_loose"] = rep["replay_matched_loose"] / rep["n_actual_trades"].replace(0, np.nan)
    rep["qty_mismatch_share_bids"] = rep["crossing_bids_qty_mismatch"] / rep["crossing_bids"].replace(0, np.nan)
    rep["qty_mismatch_share_asks"] = rep["crossing_asks_qty_mismatch"] / rep["crossing_asks"].replace(0, np.nan)
    return pr, agg, rep


def run_new_trade_analyses(df_filtered, df_all_raw, offers, output_dir, cluster_col="session_code"):
    new_dir = new_output_dir(output_dir)
    samples = {"iqr_filtered": _prep_trade_frame(df_filtered), "all_trades": _prep_trade_frame(df_all_raw)}
    tidy_all = []
    common_notes = ["Data: trades_panel.csv (executed trades, rounds 1-12 of the Carbon Trading stage, 10 sessions).",
                    "signed_diff = price - eq_price (eq_price in {3, 6, 9} is the theoretical competitive permit price = matched tax rate);",
                    "rel_dev = price / eq_price - 1; mins = seconds / 60 within the 120-second trading window.",
                    "Samples: 'iqr_filtered' = existing per-session-round IQR outlier rule of run_trade_mkt_reg.py; 'all_trades' = no filter."]

    # ---- A3: signed price deviations: descriptives and regressions ----
    desc = []
    for name, d in samples.items():
        g = d.groupby(["is_equal", "eq_price", "round_block", "sec_bin30"])
        a = g.apply(lambda x: pd.Series({"n": len(x), "mean_signed_diff": x["signed_diff"].mean(), "median_signed_diff": x["signed_diff"].median(),
                                         "qty_weighted_mean_signed_diff": np.average(x["signed_diff"], weights=x["quantity"]),
                                         "mean_rel_dev": x["rel_dev"].mean(), "median_rel_dev": x["rel_dev"].median(),
                                         "share_price_above_eq": (x["signed_diff"] > 0).mean()})).reset_index()
        a.insert(0, "sample", name)
        desc.append(a)
    pd.concat(desc, ignore_index=True).to_csv(os.path.join(new_dir, "price_signed_summary.csv"), index=False, encoding="utf-8-sig")
    f_signed = [f"signed_diff ~ is_equal*({EQ_DUMMIES}) + mins + round", f"signed_diff ~ is_equal*mins + {EQ_DUMMIES} + round",
                f"rel_dev ~ is_equal*({EQ_DUMMIES}) + mins + round", f"rel_dev ~ is_equal*mins + {EQ_DUMMIES} + round"]
    hyp_signed = {f_signed[0]: _alloc_by_price_hypotheses(), f_signed[2]: _alloc_by_price_hypotheses(),
                  f_signed[1]: ["is_equal = 0", "is_equal + 2*is_equal:mins = 0", "is_equal:mins = 0"],
                  f_signed[3]: ["is_equal = 0", "is_equal + 2*is_equal:mins = 0", "is_equal:mins = 0"]}
    for name, d in samples.items():
        tidy = new_reg_table(d, f_signed, cluster_col, os.path.join(new_dir, f"price_signed_reg_{name}.txt"),
                             f"A3. Signed price deviation from the competitive permit price ({name})",
                             common_notes + ["Hypotheses: equal-allocation effect at eq_price 6 / 3 / 9 (simple effects) and joint heterogeneity;",
                                             "in the mins models: effect at the start (mins=0) and end (mins=2) of the window."],
                             hypotheses=hyp_signed)
        tidy.insert(0, "block", f"A3_{name}"); tidy_all.append(tidy)

    # ---- A4: prices by trader-pair type ----
    pair_desc = []
    for name, d in samples.items():
        a = d.groupby(["pair_type", "is_equal", "eq_price"]).agg(n=("signed_diff", "size"), mean_signed_diff=("signed_diff", "mean"),
                                                                median_signed_diff=("signed_diff", "median"), mean_price=("price", "mean"),
                                                                qty=("quantity", "sum")).reset_index()
        a.insert(0, "sample", name)
        pair_desc.append(a)
    pd.concat(pair_desc, ignore_index=True).to_csv(os.path.join(new_dir, "price_by_pair_type.csv"), index=False, encoding="utf-8-sig")
    f_pair_fe = "signed_diff ~ C(cluster_round) + Db_Ss + Sb_Ds + Db_Ds + Db_Ss_eq + Sb_Ds_eq + Db_Ds_eq + mins"
    f_pair = f"signed_diff ~ (Db_Ss + Sb_Ds + Db_Ds)*is_equal + {EQ_DUMMIES} + mins + round"
    hyp_pair = {f_pair_fe: ["Db_Ss = 0", "Db_Ss + Db_Ss_eq = 0", "Db_Ss_eq = 0", "Sb_Ds = 0", "Sb_Ds + Sb_Ds_eq = 0", "Sb_Ds_eq = 0"],
                f_pair: ["Db_Ss = 0", "Db_Ss + Db_Ss:is_equal = 0", "Db_Ss:is_equal = 0", "Sb_Ds = 0", "Sb_Ds + Sb_Ds:is_equal = 0", "Sb_Ds:is_equal = 0"]}
    for name, d in samples.items():
        tidy = new_reg_table(d, [f_pair_fe, f_pair], cluster_col, os.path.join(new_dir, f"price_by_pair_type_reg_{name}.txt"),
                             f"A4. Signed price deviation by trader-pair type ({name})",
                             common_notes + ["Pair types: Db_Ss = dominant buyer / small seller, Sb_Ds = small buyer / dominant seller, Db_Ds = both dominant; base = small / small.",
                                             "Model (1) absorbs session x round fixed effects (C(cluster_round), not printed); *_eq terms are pair x equal-allocation interactions.",
                                             "Hypotheses: pair premium under grandfathering, under equal allocation, and their difference."],
                             hypotheses=hyp_pair)
        tidy.insert(0, "block", f"A4_{name}"); tidy_all.append(tidy)

    # ---- round-level prices (open / close / VWAP), dispersion, trade counts: all trades ----
    d_all = samples["all_trades"]
    grid = _round_grid(d_all, offers)
    rl = _round_level_prices(d_all, grid)
    rl.to_csv(os.path.join(new_dir, "price_open_close_vwap.csv"), index=False, encoding="utf-8-sig")
    rl_pos = rl[rl["n_trades"] > 0].copy()
    f_rl = [f"{y} ~ is_equal*({EQ_DUMMIES}) + round" for y in ["open_diff", "close_diff", "vwap_diff"]]
    tidy = new_reg_table(rl_pos, f_rl, cluster_col, os.path.join(new_dir, "price_open_close_vwap_reg.txt"),
                         "A4b. Opening, closing and quantity-weighted average price deviation per session-round (all trades)",
                         ["Unit: session-round with at least one trade. open/close = first/last executed trade of the round; vwap = quantity-weighted mean.",
                          "Outcome is the deviation from eq_price."], hypotheses={f: _alloc_by_price_hypotheses() for f in f_rl})
    tidy.insert(0, "block", "A4b_round_prices"); tidy_all.append(tidy)
    f_disp = [f"{y} ~ is_equal*({EQ_DUMMIES}) + round" for y in ["price_sd", "price_iqr", "price_range"]]
    tidy = new_reg_table(rl_pos, f_disp, cluster_col, os.path.join(new_dir, "price_dispersion_reg.txt"),
                         "A5. Within-round price dispersion (all trades)",
                         ["Unit: session-round with at least one trade. price_sd = population SD of executed prices; iqr and range likewise."],
                         hypotheses={f: _alloc_by_price_hypotheses() for f in f_disp})
    tidy.insert(0, "block", "A5_dispersion"); tidy_all.append(tidy)
    rl[["session_code", "round", "is_equal", "eq_price", "n_trades", "qty_traded", "price_sd", "price_iqr", "price_range", "first_trade_sec", "share_last30", "share_qty_ge2"]].to_csv(
        os.path.join(new_dir, "price_dispersion.csv"), index=False, encoding="utf-8-sig")
    dd = _dominant_dispersion(d_all)
    dd.to_csv(os.path.join(new_dir, "price_dispersion_dominant.csv"), index=False, encoding="utf-8-sig")

    # ---- A10: friction diagnostics ----
    if offers is not None and len(offers):
        pr, agg, rep = _friction_tables(offers, d_all, grid)
        agg.to_csv(os.path.join(new_dir, "offers_summary.csv"), index=False, encoding="utf-8-sig")
        qb = lambda q: np.where(q >= 3, "3+", q.astype(int).astype(str))
        oq = offers.assign(qty_bucket=qb(offers["quantity"]), side=np.where(offers["direction"] == "buy", "bid", "ask"), kind="posted_order")
        tq_b = d_all.assign(qty_bucket=qb(d_all["quantity"]), side="bid", is_dominant=d_all["buyer_is_dominant"], kind="executed_trade")
        tq_s = d_all.assign(qty_bucket=qb(d_all["quantity"]), side="ask", is_dominant=d_all["seller_is_dominant"], kind="executed_trade")
        cols = ["kind", "side", "is_dominant", "is_equal", "qty_bucket"]
        qtab = pd.concat([oq[cols], tq_b[cols], tq_s[cols]], ignore_index=True).groupby(cols).size().rename("n").reset_index()
        qtab["share_within_kind_side_type_alloc"] = qtab["n"] / qtab.groupby(["kind", "side", "is_dominant", "is_equal"])["n"].transform("sum")
        qtab.to_csv(os.path.join(new_dir, "offers_quantity_buckets.csv"), index=False, encoding="utf-8-sig")
        pr.to_csv(os.path.join(new_dir, "offers_player_round.csv"), index=False, encoding="utf-8-sig")
        fr = rep.merge(rl[["session_code", "round", "n_trades", "qty_traded", "first_trade_sec", "share_last30", "share_qty_ge2"]], on=["session_code", "round"], how="left")
        fr.to_csv(os.path.join(new_dir, "friction_diagnostics.csv"), index=False, encoding="utf-8-sig")
        f_fr = [f"n_trades ~ is_equal*({EQ_DUMMIES}) + round", f"first_trade_sec ~ is_equal*({EQ_DUMMIES}) + round",
                f"share_last30 ~ is_equal*({EQ_DUMMIES}) + round", f"qty_mismatch_share_bids ~ is_equal*({EQ_DUMMIES}) + round",
                f"qty_mismatch_share_asks ~ is_equal*({EQ_DUMMIES}) + round"]
        tidy = new_reg_table(fr.dropna(subset=["n_trades"]), f_fr[:1], cluster_col, os.path.join(new_dir, "friction_reg.txt"),
                             "A10. Trading-friction diagnostics (session-round level)",
                             ["n_trades: executed trades per round (rounds without trades included as 0).",
                              "first_trade_sec / share_last30 / qty_mismatch_share_*: only rounds with the relevant events (see friction_diagnostics.csv).",
                              "qty_mismatch_share_bids: among incoming bids that crossed the best standing ask in the order-book replay, share with no equal-quantity ask (the strict quantity-matching friction); asks likewise.",
                              "Replay validity: replay_match_share in friction_diagnostics.csv = share of actual trades reproduced by the replay (same buyer, seller, quantity, price, within 2 s);",
                              "submitted_offers records posted limit orders only, so trades executed by accepting a standing order have no matching entry, and the replay is exploratory.",
                              "Fill rates in offers_summary.csv are executed trades / posted orders per side; values above 1 mean the side traded more often by accepting standing orders than by posting."],
                             hypotheses={f_fr[0]: _alloc_by_price_hypotheses()})
        tidy.insert(0, "block", "A10_friction"); tidy_all.append(tidy)
        replay_ok = float(fr["replay_match_share"].mean()) >= 0.5
        if not replay_ok:
            print(f"[NEW] order-book replay reproduces only {fr['replay_match_share'].mean():.0%} of trades: quantity-mismatch regressions skipped (see friction_diagnostics.csv)")
        for f in f_fr[1:]:
            y = f.split("~")[0].strip()
            if y.startswith("qty_mismatch") and not replay_ok:
                continue
            sub = fr.dropna(subset=[y])
            if len(sub) > 20:
                t2 = new_reg_table(sub, [f], cluster_col, os.path.join(new_dir, f"friction_reg_{y}.txt"), f"A10. {y} (session-round level)",
                                   ["See friction_reg.txt header for definitions."], hypotheses={f: _alloc_by_price_hypotheses()})
                t2.insert(0, "block", "A10_friction"); tidy_all.append(t2)
        # dominant-firm bid fill rates (player-round level, dominant firms only)
        dom = pr[(pr["is_dominant"] == 1) & (pr["n_bids"] > 0)].copy()
        if len(dom) > 20:
            f_dom = [f"fill_rate_bids ~ is_equal*({EQ_DUMMIES}) + round", f"fill_rate_bid_qty ~ is_equal*({EQ_DUMMIES}) + round"]
            t3 = new_reg_table(dom, f_dom, cluster_col, os.path.join(new_dir, "friction_reg_dominant_bid_fill.txt"),
                               "A10. Dominant firms' bid fill rates (player-round level, rounds in which the firm submitted at least one bid)",
                               ["fill_rate_bids = executed purchases / submitted bids; fill_rate_bid_qty = quantity bought / quantity bid."],
                               hypotheses={f: _alloc_by_price_hypotheses() for f in f_dom})
            t3.insert(0, "block", "A10_friction_dominant"); tidy_all.append(t3)
    else:
        print("[NEW] offers_panel.csv not found: friction diagnostics skipped")

    pd.concat(tidy_all, ignore_index=True).to_csv(os.path.join(new_dir, "wildboot_pvalues_trade.csv"), index=False, encoding="utf-8-sig")
    print(f"[NEW] trade-market analyses written to {new_dir}")
# --- END NEW ---

# --- NEW: allocation/Hahn framing --- run the additional trade-market analyses
if RUN_NEW_ANALYSES:
	_offers_path = os.path.join(BASE_DIR, "..", "input", "offers_panel.csv")
	_offers = pd.read_csv(_offers_path) if os.path.exists(_offers_path) else None
	run_new_trade_analyses(df, df_initial_price_source, _offers, OUTPUT_DIR, cluster_col="session_code")
# --- END NEW ---
