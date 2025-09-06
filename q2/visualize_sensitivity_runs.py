#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
可视化敏感性分析的逐次运行结果（CSV）

使用示例：
  python q2/visualize_sensitivity_runs.py --csv q2/sensitivity_simulation_runs.csv \
         --outdir q2 --prefix sensitivity

输出：
  - <prefix>_bp_hist.png        每个分割点的分布直方图
  - <prefix>_week_hist.png      每组推荐孕周的分布直方图
  - <prefix>_bp_series.png      分割点随模拟序号变化的折线图
  - <prefix>_week_series.png    推荐孕周随模拟序号变化的折线图
  - <prefix>_label_changes.png  每次扰动的标签变化数量分布
控制台将打印基本统计（均值、标准差、成功率等）。
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


def _configure_chinese_font():
    """尽量自动选择可用的中文字体，避免中文方块/缺失。"""
    candidates = [
        'SimHei',                 # 黑体
        'Noto Sans CJK SC',       # 思源黑体(谷歌Noto)
        'Source Han Sans SC',     # 思源黑体
        'WenQuanYi Zen Hei',      # 文泉驿正黑
        'Microsoft YaHei',        # 微软雅黑
        'PingFang SC',            # 苹果苹方
        'STHeiti', 'Heiti SC',
        'Arial Unicode MS',
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = None
    for name in candidates:
        if name in available:
            chosen = name
            break
    if chosen is None:
        # 仍然设置一个通用无衬线并给出提示
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        print('警告：未检测到常用中文字体，可能出现中文缺字。建议安装：sudo apt-get update && sudo apt-get install -y fonts-noto-cjk')
    else:
        plt.rcParams['font.sans-serif'] = [chosen]
        plt.rcParams['axes.unicode_minus'] = False


def _sorted_numeric_cols(cols, prefix):
    def _idx(c):
        try:
            return int(c.replace(prefix, ''))
        except Exception:
            return 10**9
    return sorted([c for c in cols if c.startswith(prefix)], key=_idx)


def visualize_runs(csv_path: str, outdir: str, prefix: str) -> None:
    _configure_chinese_font()
    df = pd.read_csv(csv_path)
    if 'sim_id' in df.columns:
        df = df.sort_values('sim_id').reset_index(drop=True)

    bp_cols = _sorted_numeric_cols(df.columns, 'bp_')
    week_cols = _sorted_numeric_cols(df.columns, 'week_')

    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    # 打印基础信息
    unique_cv = sorted(df['measurement_cv'].unique()) if 'measurement_cv' in df.columns else []
    print(f"读取: {csv_path}")
    if unique_cv:
        print(f"measurement_cv 档位: {unique_cv}")
    print(f"样本条数(模拟次数): {len(df)}")
    if 'success' in df.columns:
        print(f"成功次数: {int(df['success'].sum())}/{len(df)}")

    # 统计分割点与推荐孕周
    def _print_stats(title, cols):
        if not cols:
            return
        print(f"\n{title} 统计：")
        for c in cols:
            vals = df[c].dropna().values
            if len(vals) == 0:
                continue
            print(f"  {c}: mean={np.mean(vals):.3f}, std={np.std(vals):.3f}, min={np.min(vals):.3f}, max={np.max(vals):.3f}")

    _print_stats("分割点", bp_cols)
    _print_stats("推荐孕周", week_cols)

    # 精简：仅输出“稳定性摘要”一张图 + 一份摘要CSV
    def _plot_stability_summary():
        def _col_std_list(cols):
            stats = []
            for c in cols:
                v = df[c].dropna().values
                if len(v):
                    stats.append((c, float(np.std(v))))
            return stats
        bp_std = _col_std_list(bp_cols)
        wk_std = _col_std_list(week_cols)
        avg_bp_std = float(np.mean([s for _, s in bp_std])) if bp_std else 0.0
        avg_wk_std = float(np.mean([s for _, s in wk_std])) if wk_std else 0.0
        max_bp_std = float(np.max([s for _, s in bp_std])) if bp_std else 0.0
        max_wk_std = float(np.max([s for _, s in wk_std])) if wk_std else 0.0

        if max_bp_std < 0.5 and max_wk_std < 0.5:
            stability_level = "高度稳定"
        elif max_bp_std < 1.0 and max_wk_std < 1.0:
            stability_level = "较为稳定"
        else:
            stability_level = "稳定性一般"

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 左：分割点σ条形图
        ax1 = axes[0]
        if bp_std:
            names, vals = zip(*bp_std)
            ax1.bar(range(len(vals)), vals, color='#45B7D1', alpha=0.9)
            ax1.axhline(0.5, color='orange', linestyle='--', linewidth=1.5, label='σ=0.5')
            ax1.axhline(1.0, color='red', linestyle='--', linewidth=1.2, label='σ=1.0')
            ax1.set_xticks(range(len(names)))
            ax1.set_xticklabels(names, rotation=20)
            ax1.set_ylabel('标准差 σ')
            ax1.set_title('分割点稳定性（σ）', fontsize=12, fontweight='bold')
            ax1.grid(True, axis='y', alpha=0.3)
            ax1.legend()
        else:
            ax1.axis('off')

        # 右：推荐孕周σ条形图
        ax2 = axes[1]
        if wk_std:
            names, vals = zip(*wk_std)
            ax2.bar(range(len(vals)), vals, color='#4ECDC4', alpha=0.9)
            ax2.axhline(0.5, color='orange', linestyle='--', linewidth=1.5, label='σ=0.5')
            ax2.axhline(1.0, color='red', linestyle='--', linewidth=1.2, label='σ=1.0')
            ax2.set_xticks(range(len(names)))
            ax2.set_xticklabels(names, rotation=20)
            ax2.set_ylabel('标准差 σ（周）')
            ax2.set_title('推荐孕周稳定性（σ）', fontsize=12, fontweight='bold')
            ax2.grid(True, axis='y', alpha=0.3)
            ax2.legend()
        else:
            ax2.axis('off')

        # 底部摘要文字（改变形式：给出关键结论，不再绘大量分布图）
        meta_lines = []
        if 'measurement_cv' in df.columns:
            cvs = sorted(df['measurement_cv'].dropna().unique().tolist())
            if len(cvs) == 1:
                meta_lines.append(f"测量变异系数: {cvs[0]*100:.1f}%")
            elif len(cvs) > 1:
                meta_lines.append(f"测量变异系数: {', '.join([f'{c*100:.1f}%' for c in cvs])}")
        if 'threshold_zone_width' in df.columns:
            zones = sorted(df['threshold_zone_width'].dropna().unique().tolist())
            if len(zones) == 1:
                meta_lines.append(f"关注区域: ±{zones[0]*100:.1f}%")
            elif len(zones) > 1:
                meta_lines.append(f"关注区域: ±{', '.join([f'{z*100:.1f}%' for z in zones])}")
        if 'success' in df.columns:
            meta_lines.append(f"成功次数: {int(df['success'].sum())}/{len(df)}")
        meta_lines.append(f"分割点σ: 平均 {avg_bp_std:.3f}, 最大 {max_bp_std:.3f}")
        meta_lines.append(f"推荐孕周σ: 平均 {avg_wk_std:.3f}, 最大 {max_wk_std:.3f}")
        meta_lines.append(f"总体稳定性评估: {stability_level}")

        fig.suptitle('检测误差影响的稳定性摘要', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0.08, 1, 0.95])
        fig.text(0.02, 0.02, "\n".join(meta_lines), fontsize=11,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F0F0", alpha=0.9))
        out_path = os.path.join(outdir, f"{prefix}_stability_summary.png")
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"已保存: {out_path}")

        # 精简摘要CSV
        try:
            rows = []
            for n, s in bp_std:
                rows.append({'type': 'breakpoint', 'name': n, 'std': s})
            for n, s in wk_std:
                rows.append({'type': 'week', 'name': n, 'std': s})
            rows.append({'type': 'global', 'name': 'avg_bp_std', 'value': avg_bp_std})
            rows.append({'type': 'global', 'name': 'avg_week_std', 'value': avg_wk_std})
            rows.append({'type': 'global', 'name': 'max_bp_std', 'value': max_bp_std})
            rows.append({'type': 'global', 'name': 'max_week_std', 'value': max_wk_std})
            rows.append({'type': 'global', 'name': 'stability_level', 'value': stability_level})
            if 'measurement_cv' in df.columns:
                rows.append({'type': 'meta', 'name': 'measurement_cv_unique', 'value': ','.join([str(x) for x in sorted(df['measurement_cv'].dropna().unique())])})
            if 'threshold_zone_width' in df.columns:
                rows.append({'type': 'meta', 'name': 'threshold_zone_width_unique', 'value': ','.join([str(x) for x in sorted(df['threshold_zone_width'].dropna().unique())])})
            df_out = pd.DataFrame(rows)
            csv_out = os.path.join(outdir, f"{prefix}_stability_summary.csv")
            df_out.to_csv(csv_out, index=False, encoding='utf-8-sig')
            print(f"已保存: {csv_out}")
        except Exception as e:
            print(f"保存摘要CSV失败: {e}")

    _plot_stability_summary()


def main():
    parser = argparse.ArgumentParser(description='可视化敏感性分析逐次运行结果（CSV）')
    parser.add_argument('--csv', required=True, help='CSV 文件路径，如 q2/sensitivity_simulation_runs.csv')
    parser.add_argument('--outdir', default='q2', help='输出目录')
    parser.add_argument('--prefix', default='sensitivity', help='输出文件名前缀')
    args = parser.parse_args()

    visualize_runs(args.csv, args.outdir, args.prefix)


if __name__ == '__main__':
    main()


