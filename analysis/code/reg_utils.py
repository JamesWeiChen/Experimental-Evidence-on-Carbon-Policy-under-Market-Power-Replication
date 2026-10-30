import pandas as pd
import statsmodels.formula.api as smf
from tabulate import tabulate
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os

def get_stars(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return ''

def plot_regression_results(results, model_names, y_var, cluster_info, output_path,
                            benchmark_results=None, benchmark_label="Theory benchmark"):
    """
    繪製回歸結果的係數圖表
    
    Parameters:
    results: list of regression results
    model_names: list of model names
    y_var: dependent variable name
    cluster_info: string describing clustering method
    output_path: path to save the plot
    """
    # 使用更豐富的顏色方案（從附圖中提取）
    colors = ['#51ABDD', '#227D51', '#B5495B', '#FF6F61', '#9ACD32', '#87CEEB', 
              '#006400', '#DA70D6', '#4169E1', '#A0522D', '#008B8B', '#800080']
    
    # 收集所有變數名稱（包含截距項 Intercept）
    all_vars = []
    combined_results = list(results)
    if benchmark_results:
        combined_results.extend(benchmark_results)
    for res in combined_results:
        for var in res.params.index:
            if var not in all_vars:
                all_vars.append(var)
    
    if not all_vars:
        return  # 沒有變數可以繪製
    
    # 準備資料
    coefficients = []
    std_errors = []
    p_values = []
    benchmark_coefficients = []
    
    for res in results:
        coef_list = []
        se_list = []
        p_list = []
        for var in all_vars:
            if var in res.params.index:
                coef_list.append(res.params[var])
                se_list.append(res.bse[var])
                p_list.append(res.pvalues[var])
            else:
                coef_list.append(np.nan)
                se_list.append(np.nan)
                p_list.append(np.nan)
        coefficients.append(coef_list)
        std_errors.append(se_list)
        p_values.append(p_list)

    if benchmark_results:
        for res in benchmark_results:
            coef_list = []
            for var in all_vars:
                if var in res.params.index:
                    coef_list.append(res.params[var])
                else:
                    coef_list.append(np.nan)
            benchmark_coefficients.append(coef_list)
    
    # 設定圖表參數
    x = np.arange(len(all_vars))
    # 每個變數群組總寬度（留點左右邊界）
    group_total_width = 0.8
    # 單一模型理論寬度（模型之間的中心點間距）
    bar_slot_width = group_total_width / max(len(results), 1)
    # 實際畫柱子的寬度，稍微縮小來產生模型之間的小間隔
    width = bar_slot_width * 0.85  # 15% 的間隔
    
    # 計算y軸範圍
    all_coefs = [c for coef_list in coefficients for c in coef_list if not np.isnan(c)]
    if benchmark_coefficients:
        all_coefs.extend([c for coef_list in benchmark_coefficients for c in coef_list if not np.isnan(c)])
    all_ses = [s for se_list in std_errors for s in se_list if not np.isnan(s)]
    
    if not all_coefs:
        return
    
    max_coef = max(all_coefs)
    min_coef = min(all_coefs)
    max_se = max(all_ses) if all_ses else 0
    
    # 為信賴區間留出空間
    y_range = max_coef - min_coef
    y_max = max_coef + max_se + 0.2 * y_range
    y_min = min_coef - max_se - 0.2 * y_range
    
    # 檢查是否需要截斷圖形（更嚴格的條件）
    coef_range = max_coef - min_coef
    total_range = y_max - y_min
    # 使用幅度比率並避免除以零
    denom = abs(min_coef) if abs(min_coef) > 1e-9 else 1e-9
    spread_ratio = abs(max_coef) / denom
    need_break = (coef_range > 10 and total_range > 20) or (coef_range > 5 and total_range > 15 and spread_ratio > 10)
    
    if need_break:
        # 使用截斷圖形（16:9 寬螢幕比例）
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12.8, 7.2))
        
        # 設定截斷範圍，確保 y=0 在其中一個子圖中
        # 如果所有係數都是負的，確保上方子圖包含 0
        # 如果所有係數都是正的，確保下方子圖包含 0
        # 如果有正有負，將 0 放在適當的子圖中
        if max_coef < 0:
            # 所有係數都是負的，上方子圖顯示接近 0 的區域
            break_point = min_coef + 0.3 * coef_range
            ax1.set_ylim(break_point, max(0, y_max))
            ax2.set_ylim(y_min, break_point - 0.5)
        elif min_coef > 0:
            # 所有係數都是正的，下方子圖顯示接近 0 的區域
            break_point = max_coef - 0.3 * coef_range
            ax1.set_ylim(break_point, y_max)
            ax2.set_ylim(min(0, y_min), break_point - 0.5)
        else:
            # 有正有負，根據數據分布決定截斷點
            # 確保 0 在下方子圖中
            break_point = max(5, abs(min_coef) * 0.3)  # 在正值區域設置截斷點
            ax1.set_ylim(break_point, y_max)
            ax2.set_ylim(y_min, break_point - 0.5)
        
        # 隱藏連接部分的軸線
        ax1.spines['bottom'].set_visible(False)
        ax1.tick_params(axis='x', which='both', bottom=False)
        ax2.spines['top'].set_visible(False)
        
        axes = [ax1, ax2]
        
        # 添加波浪線表示截斷
        d = 0.015
        kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
        ax1.plot((-d, +d), (-d, +d), **kwargs)
        ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs.update(transform=ax2.transAxes)
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
        
    else:
        # 使用普通圖形（16:9 寬螢幕比例）
        fig, ax = plt.subplots(figsize=(12.8, 7.2))
        ax.set_ylim(y_min, y_max)
        axes = [ax]
    
    # 繪製柱狀圖
    error_kw = dict(ecolor='lightgray', lw=1, capsize=5, capthick=1)
    rects_list = []
    
    for i, (coefs, ses, model_name) in enumerate(zip(coefficients, std_errors, model_names)):
        color = colors[i % len(colors)]
        # 使用 slot 寬度來安排柱子的中心位置，確保柱子之間有間隔
        x_pos = x + (i - (len(results) - 1) / 2) * bar_slot_width
        
        for ax in axes:
            rects = ax.bar(x_pos, coefs, width, label=f"Model {i+1}", 
                          yerr=ses, capsize=5, color=color, error_kw=error_kw)
            if ax == axes[0]:  # 只在第一個軸上收集 rects
                rects_list.append((rects, p_values[i]))

    if benchmark_coefficients:
        benchmark_color = '#D62728'
        for i, coefs in enumerate(benchmark_coefficients):
            x_pos = x + (i - (len(results) - 1) / 2) * bar_slot_width
            valid_points = [(xp, coef) for xp, coef in zip(x_pos, coefs) if not np.isnan(coef)]
            if not valid_points:
                continue
            xs, ys = zip(*valid_points)
            for ax in axes:
                ax.plot(
                    xs,
                    ys,
                    linestyle='None',
                    marker='_',
                    color=benchmark_color,
                    markersize=18,
                    markeredgewidth=2.4,
                    zorder=5,
                )
    
    # 設定標籤和標題（放大字體）
    main_ax = axes[0] if need_break else ax
    main_ax.set_ylabel('Coefficients', fontsize=16)
    title = f'Y = {y_var}'
    if cluster_info:
        title += f' | {cluster_info}'
    main_ax.set_title(title, fontsize=18)
    
    bottom_ax = axes[-1]
    bottom_ax.set_xticks(x)
    # 讓 Intercept 清楚可見
    xticklabels = [('Intercept' if v == 'Intercept' else v) for v in all_vars]
    bottom_ax.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize=14)
    
    # 放大 y 軸刻度標籤
    for ax in axes:
        ax.tick_params(axis='y', labelsize=14)
    
    # 取消圖例顯示（依使用者需求）
    # main_ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.14), ncol=min(len(results), 6), frameon=False)
    if benchmark_coefficients:
        benchmark_handle = Line2D(
            [0], [0],
            linestyle='None',
            marker='_',
            color='#D62728',
            markersize=18,
            markeredgewidth=2.4,
            label=benchmark_label
        )
        main_ax.legend(handles=[benchmark_handle], loc='upper left', frameon=False, fontsize=12)
    
    # 移除上方和右方的軸線，並添加 y=0 參考線
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # 檢查 y=0 是否在此軸的範圍內
        y_min_ax, y_max_ax = ax.get_ylim()
        if y_min_ax <= 0 <= y_max_ax:
            ax.axhline(0, color='black', linewidth=1.5, linestyle='-', zorder=3)
    
    # 添加顯著性星號
    def add_significance_stars(rects, p_vals, ax):
        for rect, p_val in zip(rects, p_vals):
            if not np.isnan(p_val):
                height = rect.get_height()
                significance = get_stars(p_val)
                if significance:
                    y_offset = 3 if height >= 0 else -18
                    ax.annotate(significance,
                               xy=(rect.get_x() + rect.get_width() / 2, height),
                               xytext=(0, y_offset),
                               textcoords="offset points",
                               ha='center', va='bottom' if height >= 0 else 'top')
    
    for rects, p_vals in rects_list:
        for ax in axes:
            add_significance_stars(rects, p_vals, ax)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[PLOT] 圖表已儲存至: {output_path}")

def custom_summary_table(results, model_names):
    first_vars = list(results[0].params.index)
    all_vars = list(first_vars)
    for res in results:
        for v in res.params.index:
            if v not in all_vars:
                all_vars.append(v)
    table = []
    for var in all_vars:
        row = [var]
        for res in results:
            if var in res.params.index:
                est = res.params[var]
                p = res.pvalues[var]
                stars = get_stars(p)
                row.append(f"{est:.3f}{stars}")
            else:
                row.append("")
        table.append(row)
        se_row = [""]
        for res in results:
            if var in res.bse.index:
                se = res.bse[var]
                se_row.append(f"({se:.3f})")
            else:
                se_row.append("")
        table.append(se_row)
    info_keys = ["N", "AIC", "BIC"]
    info_dicts = []
    for res in results:
        info = {
            "N": int(res.nobs),
            "AIC": f"{res.aic:.1f}",
            "BIC": f"{res.bic:.1f}"
        }
        info_dicts.append(info)
    for key in info_keys:
        row = [key]
        for info in info_dicts:
            row.append(str(info[key]))
        table.append(row)
    return table, all_vars


def swap_formula_y(formula, new_y):
    _, rhs = formula.split("~", 1)
    return f"{new_y} ~ {rhs.strip()}"


def reg_func(df, y_to_formulas, output_prefix, cluster_col=None, extra_effects_func=None,
             benchmark_y_map=None, benchmark_label="Theory benchmark", y_to_df=None):
    txt_outfile = output_prefix + "_reg_results.txt"
    tab_outfile = output_prefix + "_reg_results_tabulate.txt"
    
    # 設定輸出路徑
    output_dir = os.path.dirname(output_prefix)
    # 尋找專案根目錄（含有 analysis 與 doc 的那層）
    # [Replication package] 只檢查 analysis/ 的上一層（package 根目錄），不再往上搜尋；
    # doc/ 不在 package 內時，略過複製到 doc/input
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(output_dir)))
    doc_dir = None
    if os.path.isdir(os.path.join(project_root, "analysis")) and os.path.isdir(os.path.join(project_root, "doc")):
        doc_dir = os.path.join(project_root, "doc", "input")
    
    with open(txt_outfile, "w", encoding="utf-8") as f_txt, open(tab_outfile, "w", encoding="utf-8") as f_tab:
        for y, formulas in y_to_formulas.items():
            base_df = y_to_df.get(y, df) if y_to_df is not None else df
            hc3_results, cluster_results, names = [], [], []
            wald_tests_hc3, wald_tests_cluster = [], []
            benchmark_y = benchmark_y_map.get(y) if benchmark_y_map else None
            hc3_benchmark_results = [] if benchmark_y else None
            cluster_benchmark_results = [] if benchmark_y else None
            for i, fml in enumerate(formulas, 1):
                print(f"[REG] {y} model {i}: {fml}")
                names.append(f"({i})")
                # 決定子樣本
                if ("tax_rate_low" in fml or "tax_rate_high" in fml):
                    if "is_tax" in base_df.columns:
                        sub_df = base_df[base_df["is_tax"] == 1].copy()
                    else:
                        sub_df = base_df.copy()
                elif ("eq_price_low" in fml or "eq_price_high" in fml):
                    if "is_trade" in base_df.columns:
                        sub_df = base_df[base_df["is_trade"] == 1].copy()
                    else:
                        sub_df = base_df.copy()  # trade 資料本身就都是交易制度
                else:
                    sub_df = base_df.copy()
                res_hc3 = smf.ols(fml, data=sub_df).fit(cov_type="HC3")
                hc3_results.append(res_hc3)
                if benchmark_y:
                    benchmark_fml = swap_formula_y(fml, benchmark_y)
                    benchmark_res_hc3 = smf.ols(benchmark_fml, data=sub_df).fit(cov_type="HC3")
                    hc3_benchmark_results.append(benchmark_res_hc3)
                # subject-level cluster: group by cluster_col (if provided)
                if cluster_col is not None:
                    if cluster_col not in sub_df.columns:
                        raise ValueError(f"'{cluster_col}' column required for clustering.")
                    res_cluster = smf.ols(fml, data=sub_df).fit(cov_type="cluster", cov_kwds={"groups": sub_df[cluster_col]})
                    cluster_results.append(res_cluster)
                    if benchmark_y:
                        benchmark_res_cluster = smf.ols(
                            benchmark_fml,
                            data=sub_df
                        ).fit(cov_type="cluster", cov_kwds={"groups": sub_df[cluster_col]})
                        cluster_benchmark_results.append(benchmark_res_cluster)
                # Wald tests:
                if all(x in res_hc3.params.index for x in ["is_tax", "is_trade"]):
                    wald1_hc3 = res_hc3.wald_test('is_tax = is_trade')
                    wald1_cluster = res_cluster.wald_test('is_tax = is_trade') if cluster_col is not None else None
                else:
                    wald1_hc3 = None
                    wald1_cluster = None
                if all(x in res_hc3.params.index for x in ["is_tax", "is_trade", "is_tax:is_dominant", "is_trade:is_dominant"]):
                    wald2_hc3 = res_hc3.wald_test('is_tax + is_tax:is_dominant = is_trade + is_trade:is_dominant')
                    wald2_cluster = res_cluster.wald_test('is_tax + is_tax:is_dominant = is_trade + is_trade:is_dominant') if cluster_col is not None else None
                else:
                    wald2_hc3 = None
                    wald2_cluster = None
                wald_tests_hc3.append((i, wald1_hc3, wald2_hc3))
                if cluster_col is not None:
                    wald_tests_cluster.append((i, wald1_cluster, wald2_cluster))
            
            # 繪製圖表
            base_filename = os.path.basename(output_prefix)
            
            for label, results, wald_tests in [
                ("HC3 Robust SE", hc3_results, wald_tests_hc3),
                (f"Clustered SE (by {cluster_col})", cluster_results, wald_tests_cluster) if cluster_col is not None else (None, None, None)
            ]:
                if results is None:
                    continue

                if benchmark_y:
                    benchmark_results = (
                        hc3_benchmark_results
                        if label == "HC3 Robust SE"
                        else cluster_benchmark_results
                    )
                else:
                    benchmark_results = None
                
                # 生成圖表
                plot_filename = f"{base_filename}_{y}_{label.replace(' ', '_').replace('(', '').replace(')', '')}.png"
                plot_path = os.path.join(output_dir, plot_filename)
                plot_regression_results(
                    results,
                    names,
                    y,
                    label,
                    plot_path,
                    benchmark_results=benchmark_results,
                    benchmark_label=benchmark_label,
                )
                
                # 複製圖表到 doc/input
                if doc_dir is not None:
                    doc_plot_path = os.path.join(doc_dir, plot_filename)
                    try:
                        import shutil
                        shutil.copy(plot_path, doc_plot_path)
                    except Exception as e:
                        print(f"Warning: Could not copy plot to doc/input: {e}")
                
                table, all_vars = custom_summary_table(results, names)
                # Write txt (aligned)
                f_txt.write(f"=== Outcome: {y} | {label} ===\n")
                col_widths = [max(len(str(cell)) for cell in col) for col in zip(*table)]
                header = [f"Y = {y}"] + names
                f_txt.write("  ".join(h.rjust(w) for h, w in zip(header, col_widths)) + "\n")
                for row in table:
                    f_txt.write("  ".join(str(cell).rjust(w) for cell, w in zip(row, col_widths)) + "\n")
                f_txt.write("\n")
                for idx, wald1, wald2 in wald_tests:
                    if wald1 is not None:
                        f_txt.write(f"[Model {idx}] Wald test: is_tax == is_trade:  chi2={float(wald1.statistic):.3f}, p={wald1.pvalue:.4f}\n")
                    if wald2 is not None:
                        f_txt.write(f"[Model {idx}] Wald test: is_tax+is_tax:is_dominant == is_trade+is_trade:is_dominant:  chi2={float(wald2.statistic):.3f}, p={wald2.pvalue:.4f}\n")
                # Simple effects/Wald (extra)
                if extra_effects_func is not None:
                    for idx, res in enumerate(results):
                        extra_lines = extra_effects_func(res, idx, label)
                        if extra_lines:
                            f_txt.write(f"--- Extra effects for model {names[idx]} ({label}) ---\n")
                            for line in extra_lines:
                                f_txt.write(line + "\n")
                            f_txt.write("\n")
                f_txt.write("\n")
                # Write tabulate txt
                f_tab.write(f"=== Outcome: {y} | {label} ===\n")
                f_tab.write(tabulate(table, headers=[f"Y = {y}"] + names, tablefmt="github", showindex=False) + "\n\n")
                for idx, wald1, wald2 in wald_tests:
                    if wald1 is not None:
                        f_tab.write(f"[Model {idx}] Wald test: is_tax == is_trade:  chi2={float(wald1.statistic):.3f}, p={wald1.pvalue:.4f}\n")
                    if wald2 is not None:
                        f_tab.write(f"[Model {idx}] Wald test: is_tax+is_tax:is_dominant == is_trade+is_trade:is_dominant:  chi2={float(wald2.statistic):.3f}, p={wald2.pvalue:.4f}\n")
                # Simple effects/Wald (extra)
                if extra_effects_func is not None:
                    for idx, res in enumerate(results):
                        extra_lines = extra_effects_func(res, idx, label)
                        if extra_lines:
                            f_tab.write(f"--- Extra effects for model {names[idx]} ({label}) ---\n")
                            for line in extra_lines:
                                f_tab.write(line + "\n")
                            f_tab.write("\n")
                f_tab.write("\n")
        f_txt.write("Standard errors in parentheses.\n")
        f_txt.write("* p<.05, ** p<.01, *** p<.001.\n\n")
        f_tab.write("Standard errors in parentheses.\n")
        f_tab.write("* p<.05, ** p<.01, *** p<.001.\n\n")
    
    # 複製文字結果到 doc/input
    if doc_dir is not None:
        try:
            import shutil
            doc_txt_path = os.path.join(doc_dir, os.path.basename(txt_outfile))
            doc_tab_path = os.path.join(doc_dir, os.path.basename(tab_outfile))
            shutil.copy(txt_outfile, doc_txt_path)
            shutil.copy(tab_outfile, doc_tab_path)
        except Exception as e:
            print(f"Warning: Could not copy results to doc/input: {e}")

    print(f"[DONE] 結果已輸出至: {txt_outfile}\nTabulate: {tab_outfile}")
    print(f"[PLOTS] 圖表已儲存至: {output_dir}")
    if doc_dir is not None:
        print(f"[COPY] 結果已複製至: {doc_dir}")


# --- NEW: allocation/Hahn framing ---
# Helpers for the additional analyses (outputs live in analysis/output/new/).
import itertools as _itertools
import re
from scipy import stats as _sps

NEW_OUTPUT_SUBDIR = "new"
WCB_B = 9999
WCB_SEED = 20260913
WCB_WEIGHTS = "webb"


def new_output_dir(output_dir):
    """Return (and create) the folder that holds every new-analysis output."""
    path = os.path.join(str(output_dir), NEW_OUTPUT_SUBDIR)
    os.makedirs(path, exist_ok=True)
    return path


def write_text_header(fh, title, lines):
    """Documentation header at the top of a new-analysis text output."""
    fh.write(f"# {title}\n")
    for line in lines:
        fh.write(f"# {line}\n")
    fh.write("\n")


def _wcb_weight_matrix(G, B, weights, rng):
    """Bootstrap weight draws (B x G). Rademacher is enumerated exactly when G <= 12."""
    if weights == "rademacher":
        if G <= 12:
            return np.array(list(_itertools.product([-1.0, 1.0], repeat=G))), "rademacher-enumerated"
        return rng.choice([-1.0, 1.0], size=(B, G)), "rademacher"
    vals = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
    return rng.choice(vals, size=(B, G)), "webb"


def _normalize_hypothesis(hypothesis, param_names):
    """Rewrite interaction names in a constraint string to the order the fitted model uses (a:b vs b:a)."""
    names = set(param_names)
    def fix(tok):
        t = tok.group(0)
        if t in names or ":" not in t:
            return t
        parts = t.split(":")
        for perm in _itertools.permutations(parts):
            cand = ":".join(perm)
            if cand in names:
                return cand
        return t
    return re.sub(r"[A-Za-z_][\w\.\(\)\[\]',:]*", fix, hypothesis)


def _linear_constraint_matrix(hypothesis, param_names):
    """Parse a constraint string into (R, r) with R b = r, using statsmodels' own parser."""
    hypothesis = _normalize_hypothesis(hypothesis, param_names)
    try:  # statsmodels >= 0.15 (formulaic backend)
        from statsmodels.formula._manager import FormulaManager
        lc = FormulaManager().get_linear_constraints(hypothesis, param_names)
        R = np.atleast_2d(np.asarray(lc.constraint_matrix, dtype=float))
        r = np.asarray(lc.constraint_values, dtype=float).ravel()
    except Exception:  # older statsmodels (patsy backend)
        from patsy import DesignInfo
        lc = DesignInfo(param_names).linear_constraint(hypothesis)
        R = np.atleast_2d(np.asarray(lc.coefs, dtype=float))
        r = np.asarray(lc.constants, dtype=float).ravel()
    return R, r


def wild_cluster_bootstrap(df, formula, cluster_col, hypotheses, B=WCB_B, seed=WCB_SEED, weights=WCB_WEIGHTS):
    """Restricted (null-imposed) wild cluster bootstrap for linear hypotheses on an OLS model.

    Cameron, Gelbach & Miller (2008) WCR bootstrap with Webb (2014) six-point weights by default
    (recommended with few clusters); Rademacher weights are enumerated exactly when G <= 12.
    Each hypothesis is a statsmodels/patsy constraint string, e.g. "is_equal = 0" or
    "is_equal + is_equal:eq_price_low = 0". Single restrictions use the cluster-robust t statistic,
    joint restrictions the cluster-robust Wald statistic. The bootstrap p-value is the share of
    bootstrap statistics at least as extreme as the sample statistic (symmetric for t).

    Returns a list of dicts (one per hypothesis) with the estimate, cluster SE, cluster p-values
    (normal reference as in the existing tables, and t(G-1) reference), and the bootstrap p-value.
    """
    model = smf.ols(formula, data=df)
    res_c = model.fit(cov_type="cluster", cov_kwds={"groups": df.loc[model.data.row_labels, cluster_col]})
    X = np.asarray(model.exog, dtype=float)
    y = np.asarray(model.endog, dtype=float)
    N, k = X.shape
    codes, uniq = pd.factorize(df.loc[model.data.row_labels, cluster_col].to_numpy())
    G = len(uniq)
    groups = [np.where(codes == j)[0] for j in range(G)]
    XtX_inv = np.linalg.pinv(X.T @ X)
    b_hat = XtX_inv @ X.T @ y
    u_hat = y - X @ b_hat
    ssc = (G / (G - 1)) * ((N - 1) / (N - k))
    M = [X[idx].T @ X[idx] for idx in groups]
    rng = np.random.default_rng(seed)
    W, wlabel = _wcb_weight_matrix(G, B, weights, rng)
    param_names = list(model.data.param_names)
    out = []
    for hyp in hypotheses:
        R, r = _linear_constraint_matrix(hyp, param_names)
        q = R.shape[0]
        # restricted estimate and residuals (null imposed)
        A = R @ XtX_inv @ R.T
        b_r = b_hat - XtX_inv @ R.T @ np.linalg.solve(A, R @ b_hat - r)
        u_r = y - X @ b_r
        s_g = [X[idx].T @ u_r[idx] for idx in groups]              # cluster scores under the null
        S = XtX_inv @ np.column_stack(s_g)                           # k x G : b* = b_r + S w
        # sample statistic with unrestricted residuals
        meat = np.zeros((k, k))
        for idx in groups:
            sc = X[idx].T @ u_hat[idx]
            meat += np.outer(sc, sc)
        V = ssc * (XtX_inv @ meat @ XtX_inv)
        est = R @ b_hat - r
        RVR = R @ V @ R.T
        if q == 1:
            se = float(np.sqrt(RVR[0, 0]))
            stat = float(est[0] / se)
            # fast vectorised bootstrap for a single restriction
            a = (XtX_inv @ R.T).ravel()                              # k
            alpha = np.array([a @ s for s in s_g])                   # G
            beta = np.column_stack([Mg @ a for Mg in M])             # k x G  (beta_g = M_g a)
            T = S @ W.T                                              # k x B
            num = (R @ S).ravel() @ W.T                              # B : R (b* - b_r)
            terms = W * alpha[None, :] - (beta.T @ T).T              # B x G : a' score_g*
            var = ssc * np.sum(terms ** 2, axis=1)
            boot = num / np.sqrt(var)
            p_wcb = float(np.mean(np.abs(boot) >= abs(stat) - 1e-12))
            p_norm = float(2 * _sps.norm.sf(abs(stat)))
            p_tG = float(2 * _sps.t.sf(abs(stat), df=G - 1))
        else:
            se = np.nan
            stat = float(est @ np.linalg.solve(RVR, est))
            boot = np.empty(len(W))
            for bi, w in enumerate(W):
                bstar = b_r + S @ w
                ustar = y - X @ bstar
                meat_b = np.zeros((k, k))
                for idx in groups:
                    sc = X[idx].T @ ustar[idx]
                    meat_b += np.outer(sc, sc)
                Vb = ssc * (XtX_inv @ meat_b @ XtX_inv)
                e = R @ bstar - r
                boot[bi] = e @ np.linalg.solve(R @ Vb @ R.T, e)
            p_wcb = float(np.mean(boot >= stat - 1e-12))
            p_norm = float(_sps.chi2.sf(stat, df=q))
            p_tG = float(_sps.f.sf(stat / q, q, G - 1))
        out.append({
            "hypothesis": hyp, "estimate": float(est[0]) if q == 1 else np.nan,
            "se_cluster": se, "stat": stat, "p_cluster_normal": p_norm, "p_cluster_tG1": p_tG,
            "p_wcb": p_wcb, "weights": wlabel, "B": int(len(W)), "G": int(G), "N": int(N),
        })
    return out


def new_reg_table(df, formulas, cluster_col, out_path, title, notes, hypotheses=None,
                  B=WCB_B, seed=WCB_SEED, weights=WCB_WEIGHTS, hide_prefix=("C(",)):
    """Compact regression table for the new analyses (no plots, no doc/input copies).

    Each formula is estimated by OLS with standard errors clustered at `cluster_col`; every
    coefficient gets the cluster p-value (normal reference, as in the existing tables) and a wild
    cluster bootstrap p-value. `hypotheses` maps a formula (string) to a list of constraint strings
    that are tested the same way (simple effects, differences). Coefficients whose name starts with
    any of `hide_prefix` (e.g. fixed-effect dummies) are estimated but not printed.
    Returns a tidy DataFrame with one row per (formula, term).
    """
    tidy = []
    lines_out = []
    for i, fml in enumerate(formulas, 1):
        res = smf.ols(fml, data=df).fit(cov_type="cluster", cov_kwds={"groups": df[cluster_col]})
        names = [n for n in res.params.index if not str(n).startswith(hide_prefix)]
        wcb = wild_cluster_bootstrap(df, fml, cluster_col, [f"{n} = 0" for n in names], B=B, seed=seed, weights=weights)
        G = wcb[0]["G"] if wcb else np.nan
        table = []
        for n, w in zip(names, wcb):
            table.append([n, f"{res.params[n]:.3f}{get_stars(w['p_wcb'])}", f"({res.bse[n]:.3f})",
                          f"{w['p_cluster_normal']:.3f}", f"{w['p_cluster_tG1']:.3f}", f"{w['p_wcb']:.3f}"])
            tidy.append({"formula": fml, "model": i, "term": n, "coef": float(res.params[n]), "se_cluster": float(res.bse[n]),
                         "p_cluster_normal": w["p_cluster_normal"], "p_cluster_tG1": w["p_cluster_tG1"], "p_wcb": w["p_wcb"],
                         "N": int(res.nobs), "G": G, "r2": float(res.rsquared), "weights": w["weights"], "B": w["B"]})
        lines_out.append(f"=== Model ({i}): {fml}")
        lines_out.append(f"N = {int(res.nobs)}, clusters ({cluster_col}) = {G}, R2 = {res.rsquared:.3f}, "
                         f"bootstrap = {wcb[0]['weights'] if wcb else ''} (B = {wcb[0]['B'] if wcb else ''})")
        lines_out.append(tabulate(table, headers=["term", "coef", "cluster SE", "p (cluster, normal)", "p (cluster, t G-1)", "p (wild cluster boot)"], tablefmt="github"))
        extra = (hypotheses or {}).get(fml)
        if extra:
            ex = wild_cluster_bootstrap(df, fml, cluster_col, extra, B=B, seed=seed, weights=weights)
            rows = [[h["hypothesis"], f"{h['estimate']:.3f}" if not np.isnan(h["estimate"]) else "(joint)", f"({h['se_cluster']:.3f})" if not np.isnan(h["se_cluster"]) else "",
                     f"{h['p_cluster_normal']:.3f}", f"{h['p_cluster_tG1']:.3f}", f"{h['p_wcb']:.3f}"] for h in ex]
            lines_out.append("--- linear hypotheses ---")
            lines_out.append(tabulate(rows, headers=["hypothesis", "estimate", "cluster SE", "p (cluster, normal)", "p (cluster, t G-1)", "p (wild cluster boot)"], tablefmt="github"))
            for h in ex:
                tidy.append({"formula": fml, "model": i, "term": h["hypothesis"], "coef": h["estimate"], "se_cluster": h["se_cluster"],
                             "p_cluster_normal": h["p_cluster_normal"], "p_cluster_tG1": h["p_cluster_tG1"], "p_wcb": h["p_wcb"],
                             "N": h["N"], "G": h["G"], "r2": np.nan, "weights": h["weights"], "B": h["B"]})
        lines_out.append("")
    with open(out_path, "w", encoding="utf-8") as fh:
        write_text_header(fh, title, list(notes) + [
            f"Standard errors clustered by {cluster_col}; stars use the wild-cluster-bootstrap p-value: * p<.05, ** p<.01, *** p<.001.",
            "p (cluster, normal): statsmodels cluster-robust p with a normal reference, the convention of the existing tables.",
            "p (cluster, t G-1): same statistic with a t(G-1) reference.",
            "p (wild cluster boot): restricted wild cluster bootstrap (Cameron-Gelbach-Miller 2008), Webb six-point weights unless stated.",
        ])
        fh.write("\n".join(lines_out) + "\n")
    print(f"[NEW] regression table written: {out_path}")
    return pd.DataFrame(tidy)
# --- END NEW ---
