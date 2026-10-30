import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from scipy import stats

# --- NEW: allocation/Hahn framing ---
RUN_NEW_ANALYSES = True   # set False to skip the additional analyses written to analysis/output/new/
# --- END NEW ---

# 參數與路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT = os.path.join(BASE_DIR, "..", "input", "trades_panel.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "output")
# doc 跟 analysis 是平行資料夾
ANALYSIS_DIR = os.path.dirname(BASE_DIR)  # 從 analysis/code 到 analysis
PROJECT_DIR = os.path.dirname(ANALYSIS_DIR)  # 從 analysis 到上一層
DOC_INPUT_DIR = os.path.join(PROJECT_DIR, "doc", "input")
os.makedirs(OUTPUT_DIR, exist_ok=True)
# [Replication package] doc/ 不在 package 內時，略過複製到 doc/input

# 讀資料
print(f"Looking for input file: {INPUT}")
if not os.path.exists(INPUT):
    raise FileNotFoundError("trades_panel.csv not found in input directory.")
print("Loading data...")
df = pd.read_csv(INPUT)
print(f"Data loaded: {len(df)} rows")
df["price_diff"] = df["price"] - df["eq_price"]

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
print("  (異常值判定標準：在每個 session-round 內使用 Q1-1.5×IQR 和 Q3+1.5×IQR)")

print("\nStarting plot generation...")


# 顏色: is_equal+eq_price, 形狀: eq_price, 線型: eq_price
# is_equal=0: 冷色系（淺綠、藍、深藍），is_equal=1: 暖色系（淺橘、橘、深紅）
color_map = {
    (0, 3): "#9ACD32",   # 淺綠
    (0, 6): "#008B8B",   # 藍
    (0, 9): "#006400",   # 深綠
    (1, 3): "#FD8D3C",   # 淺橘
    (1, 6): "#FF6F61",   # 橘
    (1, 9): "#B30000",   # 深紅
}
eq_price_markers = {3: "o", 6: "o", 9: "o"}
eq_price_linestyles = {3: "solid", 6: "solid", 9: "solid"}
levels = [3, 6, 9]

plot_df = df[["seconds", "price_diff", "eq_price", "is_equal", "round"]].dropna()

# 創建10秒binning函數
def bin_data_by_10_seconds(data):
    """將數據按10秒為單位進行分組，計算每個bin的中位數price_diff和樣本數量"""
    if data.empty:
        return pd.DataFrame()
        
    binned_data = []
    
    for ie in [0, 1]:
        for lvl in levels:
            g = data[(data["is_equal"] == ie) & (data["eq_price"] == lvl)]
            if g.empty:
                continue
            
            # 創建10秒bins
            min_sec = int(g["seconds"].min() // 10) * 10
            max_sec = int(g["seconds"].max() // 10 + 1) * 10
            bins = np.arange(min_sec, max_sec + 10, 10)
            
            # 為每個數據點分配bin
            g = g.copy()
            g["bin"] = ((g["seconds"] // 10) * 10).astype(int)
            
            # 計算每個bin的統計數據（使用中位數）
            bin_stats = g.groupby("bin").agg({
                "price_diff": "median",  # 改用中位數
                "seconds": "count"  # 樣本數量
            }).reset_index()
            bin_stats.rename(columns={"seconds": "sample_count"}, inplace=True)
            
            # 添加分組信息
            bin_stats["is_equal"] = ie
            bin_stats["eq_price"] = lvl
            bin_stats["bin_center"] = bin_stats["bin"] + 5  # bin中心點
            
            binned_data.append(bin_stats)
    
    if binned_data:
        return pd.concat(binned_data, ignore_index=True)
    else:
        return pd.DataFrame()


def plot_seconds_vs_price_diff(data, title, filename_base, plot_only_equal_zero=False):
    """通用繪圖函數，用於生成 seconds vs price_diff 的散點圖"""
    plt.figure(figsize=(9, 6))

    if plot_only_equal_zero:
        plot_data = data[data["is_equal"] == 0]
        is_equal_values = [0]
    else:
        plot_data = data
        is_equal_values = [1, 0]

    binned_data = bin_data_by_10_seconds(plot_data)

    if binned_data.empty:
        print(f"Skipping plot for '{title}' due to empty data.")
        plt.close()
        return

    min_count = binned_data["sample_count"].min()
    max_count = binned_data["sample_count"].max()
    min_size = 20
    max_size = 200
    
    legend_handles = []

    for ie in is_equal_values:
        for lvl in levels:
            g_binned = binned_data[(binned_data["is_equal"] == ie) & (binned_data["eq_price"] == lvl)]
            g_raw = plot_data[(plot_data["is_equal"] == ie) & (plot_data["eq_price"] == lvl)]
            
            if g_binned.empty or g_raw.empty:
                continue

            # 繪製binned scatter points
            x = g_binned["bin_center"].to_numpy()
            y = g_binned["price_diff"].to_numpy()
            counts = g_binned["sample_count"].to_numpy()

            if max_count > min_count:
                sizes = min_size + (max_size - min_size) * (counts - min_count) / (max_count - min_count)
            else:
                sizes = np.full_like(counts, min_size)

            color = color_map[(ie, lvl)]
            marker = eq_price_markers[lvl]
            linestyle = eq_price_linestyles[lvl]
            plt.scatter(x, y, alpha=0.65, s=sizes, c=[color], marker=marker, linewidths=0)

            # 使用原始數據擬合穩健回歸線（Theil-Sen estimator）
            x_raw = g_raw["seconds"].to_numpy()
            y_raw = g_raw["price_diff"].to_numpy()
            if len(x_raw) >= 2 and np.std(x_raw) > 0:
                # Theil-Sen 是一種穩健的線性回歸方法，不受極端值影響
                # 計算所有點對的斜率中位數
                slope, intercept, _, _ = stats.theilslopes(y_raw, x_raw)
                xs = np.linspace(x_raw.min(), x_raw.max(), 100)
                ys = intercept + slope * xs
                plt.plot(xs, ys, c=color, linestyle=linestyle, linewidth=2)
            
            label = f"is_equal={ie}, eq_price={lvl}"
            legend_handles.append(
                mlines.Line2D([], [], color=color, linestyle=linestyle, marker=marker,
                              label=label, linewidth=2, markersize=8)
            )

    plt.xlabel("seconds (10-second bins)", fontsize=16)
    plt.ylabel("price_diff = price - eq_price", fontsize=16)
    plt.ylim(0, 50)
    plt.title(title, fontsize=18)
    
    # 放大刻度標籤
    plt.tick_params(axis='both', which='major', labelsize=14)
    
    plt.legend(handles=legend_handles, loc="upper right", fontsize=12)
    
    plt.tight_layout()
    
    # 儲存並複製檔案
    output_file = os.path.join(OUTPUT_DIR, f"{filename_base}.png")
    plt.savefig(output_file, dpi=200)
    plt.close()
    
    if os.path.isdir(DOC_INPUT_DIR):
        try:
            import shutil
            doc_input_file = os.path.join(DOC_INPUT_DIR, f"{filename_base}.png")
            shutil.copy(output_file, doc_input_file)
        except Exception as e:
            print(f"Warning: Could not copy plot to doc/input: {e}")


# === 定義要生成的圖表 ===
plot_configs = [
    {
        "data": plot_df,
        "title": "seconds vs price_diff (10-second bins, size = sample count)",
        "filename_base": "trade_mkt_seconds_vs_price_diff",
        "plot_only_equal_zero": False
    },
    {
        "data": plot_df,
        "title": "seconds vs price_diff (10-second bins, size = sample count)",
        "filename_base": "trade_mkt_seconds_vs_price_diff_equal0_only",
        "plot_only_equal_zero": True
    },
    {
        "data": plot_df[(plot_df["round"] >= 4) & (plot_df["round"] <= 12)],
        "title": "seconds vs price_diff (round 4-12, 10-second bins, size = sample count)",
        "filename_base": "trade_mkt_seconds_vs_price_diff_round_4-12",
        "plot_only_equal_zero": False
    },
    {
        "data": plot_df[(plot_df["round"] >= 4) & (plot_df["round"] <= 12)],
        "title": "seconds vs price_diff (round 4-12, 10-second bins, size = sample count)",
        "filename_base": "trade_mkt_seconds_vs_price_diff_round_4-12_equal0_only",
        "plot_only_equal_zero": True
    },
    {
        "data": plot_df[(plot_df["round"] >= 7) & (plot_df["round"] <= 12)],
        "title": "seconds vs price_diff (round 7-12, 10-second bins, size = sample count)",
        "filename_base": "trade_mkt_seconds_vs_price_diff_round_7-12",
        "plot_only_equal_zero": False
    },
    {
        "data": plot_df[(plot_df["round"] >= 7) & (plot_df["round"] <= 12)],
        "title": "seconds vs price_diff (round 7-12, 10-second bins, size = sample count)",
        "filename_base": "trade_mkt_seconds_vs_price_diff_round_7-12_equal0_only",
        "plot_only_equal_zero": True
    }
]

# === 循環生成所有圖表 ===
for config in plot_configs:
    print(f"Generating plot: {config['title']} ({'is_equal=0 only' if config['plot_only_equal_zero'] else 'all data'})")
    plot_seconds_vs_price_diff(
        data=config["data"],
        title=config["title"],
        filename_base=config["filename_base"],
        plot_only_equal_zero=config["plot_only_equal_zero"]
    )

print("\nAll scatter plots generated and copied to doc/input successfully!")
print(f"Generated files: {[f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png') and 'trade_mkt_seconds' in f]}")
print("Script completed successfully!")

# --- NEW: allocation/Hahn framing ---
# Signed price deviation (price - eq_price) by 10-second bin and allocation, one panel per eq_price,
# mean with 95% CI band and median; IQR-filtered trades as above. Saved to analysis/output/new/.
if RUN_NEW_ANALYSES:
    import sys as _sys
    _sys.path.append(BASE_DIR)
    from reg_utils import new_output_dir
    _new_dir = new_output_dir(OUTPUT_DIR)
    _d = plot_df.copy()
    _d["bin"] = (_d["seconds"] // 10 * 10).astype(int) + 5
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, lvl in zip(axes, levels):
        for ie, col, lab in ((0, "#227D51", "grandfathering"), (1, "#51ABDD", "equal")):
            g = _d[(_d["is_equal"] == ie) & (_d["eq_price"] == lvl)].groupby("bin")["price_diff"]
            st = g.agg(["mean", "median", "std", "count"]).reset_index()
            st["ci"] = 1.96 * st["std"] / np.sqrt(st["count"].clip(lower=1))
            ax.plot(st["bin"], st["mean"], color=col, lw=2, label=f"{lab}: mean")
            ax.fill_between(st["bin"], st["mean"] - st["ci"], st["mean"] + st["ci"], color=col, alpha=0.15)
            ax.plot(st["bin"], st["median"], color=col, lw=1.2, ls="--", label=f"{lab}: median")
        ax.axhline(0, color="black", lw=1)
        ax.set_title(f"eq_price = {lvl}", fontsize=14); ax.set_xlabel("seconds (10-s bins)", fontsize=12)
        ax.tick_params(labelsize=11)
    axes[0].set_ylabel("price - eq_price", fontsize=12); axes[0].legend(fontsize=9, loc="upper right")
    fig.suptitle("Signed price deviation from the competitive permit price (IQR-filtered trades, rounds 1-12)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(_new_dir, "fig_price_signed_vs_seconds.png"), dpi=200)
    plt.close(fig)
    print(f"[NEW] figure saved: {os.path.join(_new_dir, 'fig_price_signed_vs_seconds.png')}")
# --- END NEW ---
