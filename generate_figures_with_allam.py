#!/usr/bin/env python3
"""
Generate all 7 paper figures with 5 models (original 4 + ALLaM-S3).
KEY PRINCIPLE: Original 4 models use data from the ORIGINAL CSV (table2_benchmark_results.csv)
so their shapes/values are IDENTICAL to the original figures.
ALLaM-S3 is added from benchmark_with_allam.csv.
Figures 2 and 5 remain 4-model only.
Output folder: "صور بعد اضافه علام/"
"""

import os, shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob

BASE = "/Users/areejbaw/Downloads/orkspace 3"
CSV_ORIG = os.path.join(BASE, "dataset/final_results/table2_benchmark_results.csv")
CSV_ALLAM = os.path.join(BASE, "dataset/simulation/full_experiment_300_students/figures_and_tables/benchmark_with_allam.csv")
SIM_BASE = os.path.join(BASE, "dataset/simulation/full_experiment_300_students")
OUT = os.path.join(BASE, "صور بعد اضافه علام")
os.makedirs(OUT, exist_ok=True)

# ============================================================
# Colors matching original figures exactly
# ============================================================
MODEL_COLORS = {
    'GPT-3.5':     '#E74C3C',
    'GPT-4o Mini': '#2ECC71',
    'Llama 3.1':   '#3498DB',
    'Qwen2.5 7B':  '#E67E22',
    'ALLaM-7B':    '#9B59B6',
}
MODEL_FILL = {
    'GPT-3.5':     '#FADBD8',
    'GPT-4o Mini': '#D5F5E3',
    'Llama 3.1':   '#D6EAF8',
    'Qwen2.5 7B':  '#FDEBD0',
    'ALLaM-7B':    '#E8DAEF',
}

models_4 = ['GPT-3.5', 'GPT-4o Mini', 'Llama 3.1', 'Qwen2.5 7B']
models_5 = ['GPT-3.5', 'GPT-4o Mini', 'Llama 3.1', 'Qwen2.5 7B', 'ALLaM-7B']

# ============================================================
# Load original 4-model data (EXACT same source as original figures)
# ============================================================
df_orig = pd.read_csv(CSV_ORIG)
# Column names have dots: GA., MO., WO., CU., FOC., FOL., ENG., CO., CH., AC., COR.

def get_orig4(model_name, exp):
    """Get row from original 4-model CSV."""
    if 'Qwen' in model_name:
        mask = df_orig['MODEL'].str.contains('Qwen', na=False) & (df_orig['EXP'] == exp)
    else:
        mask = (df_orig['MODEL'] == model_name) & (df_orig['EXP'] == exp)
    rows = df_orig[mask]
    if len(rows) > 0:
        return rows.iloc[0]
    return None

# Load ALLaM data
df_allam = pd.read_csv(CSV_ALLAM)

def get_allam(strategy, exp):
    """Get ALLaM row from 5-model CSV."""
    mask = df_allam['Model'].str.contains(strategy, na=False) & (df_allam['Exp'] == exp)
    rows = df_allam[mask]
    if len(rows) > 0:
        return rows.iloc[0]
    return None

# ============================================================
# Helper: get metric value for any model
# ============================================================
def get_val(model, exp, metric):
    """Get metric value from the correct CSV source."""
    if model == 'ALLaM-7B':
        row = get_allam('ALLaM S3', exp)
        if row is not None and metric in row.index:
            return float(row[metric])
        return None
    else:
        row = get_orig4(model, exp)
        if row is None:
            return None
        # Original CSV has dots in column names
        dotted = metric + '.'
        if dotted in row.index:
            return float(row[dotted])
        elif metric in row.index:
            return float(row[metric])
        # Try underscore versions
        underscore_map = {
            'GA': '_gaze_dist', 'MO': '_motor_dist', 'WO': '_workload',
            'CU': '_curiosity', 'FOC': '_focus', 'FOL': '_course_follow',
            'ENG': '_engagement', 'CO': '_confusion',
            'CH': '_choice_sim', 'AC': '_acc_sim', 'COR': '_correct'
        }
        if metric in underscore_map and underscore_map[metric] in row.index:
            return float(row[underscore_map[metric]])
        return None


# ============================================================
# Figure 1: Heatmap - Real vs Sim Consistency (5 models)
# ============================================================
def make_fig1():
    print("Generating Figure 1: Heatmap...")
    metrics_hm = ['GA', 'MO', 'WO', 'CU', 'FOC', 'FOL', 'CH', 'AC', 'COR']

    heatmap_data = []
    model_labels = []

    for m in models_5:
        row = []
        for met in metrics_hm:
            rv = get_val(m, 'Real', met)
            sv = get_val(m, 'Sim', met)
            if rv is None or sv is None:
                row.append(np.nan)
                continue
            if met in ['CH', 'AC', 'COR']:
                diff = sv - rv
            else:
                if rv != 0:
                    diff = (sv - rv) / rv * 100
                else:
                    diff = 0
            row.append(round(diff, 1))

        # If ALLaM has no AC, mark it
        heatmap_data.append(row)
        label = m if m != 'ALLaM-7B' else 'ALLaM-7B (S3)'
        model_labels.append(label)

    heatmap_data = np.array(heatmap_data)

    # Check if ALLaM row has NaN for AC — if so remove AC column
    has_ac = not np.isnan(heatmap_data[-1, metrics_hm.index('AC')])
    if not has_ac:
        ac_idx = metrics_hm.index('AC')
        heatmap_data = np.delete(heatmap_data, ac_idx, axis=1)
        metrics_hm = [m for i, m in enumerate(metrics_hm) if i != ac_idx]

    fig, ax = plt.subplots(figsize=(14, 8))
    cmap = plt.cm.RdYlGn_r
    norm = mcolors.TwoSlopeNorm(vmin=-30, vcenter=0, vmax=30)
    im = ax.imshow(heatmap_data, cmap=cmap, norm=norm, aspect='auto')

    ax.set_xticks(range(len(metrics_hm)))
    ax.set_xticklabels(metrics_hm, fontsize=12, fontweight='bold')
    ax.set_yticks(range(len(model_labels)))
    ax.set_yticklabels(model_labels, fontsize=12, fontweight='bold')

    for i in range(len(model_labels)):
        for j in range(len(metrics_hm)):
            val = heatmap_data[i, j]
            if np.isnan(val):
                continue
            color = 'white' if abs(val) > 15 else 'black'
            ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=color)

    for i in range(len(model_labels) + 1):
        ax.axhline(i - 0.5, color='white', linewidth=2)
    for j in range(len(metrics_hm) + 1):
        ax.axvline(j - 0.5, color='white', linewidth=2)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Difference % ((Sim - Real) / Real × 100)', fontsize=11)

    ax.set_xlabel('Metrics', fontsize=13, fontweight='bold')
    ax.set_ylabel('Models', fontsize=13, fontweight='bold')
    ax.set_title('Real vs Sim Consistency Analysis\nGreen = Similar (Good Consistency) | Red = Different (Poor Consistency)',
                 fontsize=14, fontweight='bold')

    avg_diffs = {}
    for i, m in enumerate(model_labels):
        valid = heatmap_data[i][~np.isnan(heatmap_data[i])]
        avg_diffs[m] = np.mean(np.abs(valid))
    best = min(avg_diffs, key=avg_diffs.get)
    worst = max(avg_diffs, key=avg_diffs.get)
    ax.text(0.5, -0.12,
            f'□ Best Consistency: {best} (avg diff: {avg_diffs[best]:.1f}%) | △ Worst: {worst} (avg diff: {avg_diffs[worst]:.1f}%)',
            transform=ax.transAxes, ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'figure1_real_vs_sim_heatmap.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Done.")


# ============================================================
# Figure 2: Arabic vs English (4 models only - copy original)
# ============================================================
def make_fig2():
    src = os.path.join(BASE, "ملف المشروع قبل التعديل", "figure2_arabic_english_gap.png")
    shutil.copy2(src, os.path.join(OUT, "figure2_arabic_english_gap.png"))
    print("Figure 2: Copied original (ALLaM excluded - no English benchmark).")


# ============================================================
# Figure 3: Radar Charts (5 models)
# Original 4 use EXACT same data → same shapes
# ============================================================
def make_fig3():
    print("Generating Figure 3: Radar Charts (5 models)...")

    # EXACT same metric order as original: WO(top), MO, GA, AC, CH, FOL, FOC, CU
    radar_labels_8 = ['WO', 'MO', 'GA', 'AC', 'CH', 'FOL', 'FOC', 'CU']
    # ALLaM has no AC
    radar_labels_7 = ['WO', 'MO', 'GA', 'CH', 'FOL', 'FOC', 'CU']

    def draw_radar(ax, model, labels):
        N = len(labels)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        real_vals = []
        sim_vals = []
        for met in labels:
            rv = get_val(model, 'Real', met)
            sv = get_val(model, 'Sim', met)
            if rv is None: rv = 0
            if sv is None: sv = 0
            if met in ['CH', 'AC', 'COR']:
                real_vals.append(rv / 100.0)
                sim_vals.append(sv / 100.0)
            else:
                real_vals.append(1 - rv)
                sim_vals.append(1 - sv)
        real_vals += real_vals[:1]
        sim_vals += sim_vals[:1]

        color = MODEL_COLORS[model]
        fill = MODEL_FILL[model]

        # Start from top (90 degrees = pi/2)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)  # clockwise

        ax.plot(angles, real_vals, 'o-', color=color, linewidth=2, label='Real')
        ax.fill(angles, real_vals, alpha=0.25, color=fill)
        ax.plot(angles, sim_vals, 's--', color='gray', linewidth=1.5, label='Sim')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 1)
        title = f'{model}\nCapability Profile' if model != 'ALLaM-7B' else 'ALLaM-7B (S3)\nCapability Profile'
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
        ax.legend(loc='upper right', fontsize=8)

    # Layout: 2x2 for original 4, then a 5th below-center
    fig = plt.figure(figsize=(16, 20))
    fig.suptitle('Model Capability Radar Charts\n(Larger area = Better overall performance)',
                 fontsize=16, fontweight='bold', y=0.97)

    # Top row: GPT-3.5, GPT-4o Mini
    ax1 = fig.add_subplot(3, 2, 1, polar=True)
    ax2 = fig.add_subplot(3, 2, 2, polar=True)
    # Middle row: Llama 3.1, Qwen2.5 7B
    ax3 = fig.add_subplot(3, 2, 3, polar=True)
    ax4 = fig.add_subplot(3, 2, 4, polar=True)
    # Bottom row: ALLaM-7B centered
    ax5 = fig.add_subplot(3, 2, (5, 6), polar=True)

    draw_radar(ax1, 'GPT-3.5', radar_labels_8)
    draw_radar(ax2, 'GPT-4o Mini', radar_labels_8)
    draw_radar(ax3, 'Llama 3.1', radar_labels_8)
    draw_radar(ax4, 'Qwen2.5 7B', radar_labels_8)
    draw_radar(ax5, 'ALLaM-7B', radar_labels_7)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUT, 'figure3_model_radar_charts.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Done.")


# ============================================================
# Figure 4: Student Difficulty Spectrum (5 models)
# ============================================================
def compute_ch_per_student(log_dir):
    """Compute choice similarity per student from individual CSV files."""
    post_files = glob.glob(os.path.join(log_dir, '*_result_ind_post.csv'))
    student_ch = {}
    for f in post_files:
        try:
            df_s = pd.read_csv(f)
            sid = df_s['student_id'].iloc[0]
            if 'choice_similarity' in df_s.columns:
                valid = pd.to_numeric(df_s['choice_similarity'], errors='coerce').dropna()
                if len(valid) > 0:
                    student_ch[sid] = valid.mean() * 100
        except:
            continue
    return student_ch

def make_fig4():
    print("Generating Figure 4: Student Difficulty Spectrum (5 models)...")

    sim_folders = {
        'GPT-3.5': 'sim_only_recent_one_reflect-no_KM+PM+MM+CM_standard_cog_example-no_2',
        'GPT-4o Mini': 'sim_only_recent_one_reflect-no_KM+PM+MM+CM_standard_cog_example-no_3',
        'Llama 3.1': 'sim_only_recent_one_reflect-no_KM+PM+MM+CM_standard_cog_example-no_4',
        'Qwen2.5 7B': 'sim_only_recent_one_reflect-no_KM+PM+MM+CM_standard_cog_example-no_5',
        'ALLaM-7B': 'sim_no_memory_reflect-no_KM_standard_example-no_allam_s3',
    }

    difficulty_bins = [
        ('Easy\n(CH>70%)',   lambda x: x > 70),
        ('Medium\n(50-70%)', lambda x: (x >= 50) & (x <= 70)),
        ('Hard\n(30-50%)',   lambda x: (x >= 30) & (x < 50)),
        ('Very Hard\n(CH<30%)', lambda x: x < 30),
    ]
    bin_colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle('Student Difficulty Spectrum Across Models\n(Based on Choice Similarity - Lower CH = Harder to Simulate)',
                 fontsize=14, fontweight='bold')
    axes_flat = axes.flatten()

    for idx, model in enumerate(models_5):
        ax = axes_flat[idx]
        folder = sim_folders[model]
        log_dir = os.path.join(SIM_BASE, folder, 'log')
        ch_dict = compute_ch_per_student(log_dir)
        ch_values = np.array(list(ch_dict.values()))

        if len(ch_values) == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue

        counts = []
        labels = []
        for bname, bfunc in difficulty_bins:
            counts.append(int(np.sum(bfunc(ch_values))))
            labels.append(bname)

        total = len(ch_values)
        y_pos = list(range(len(labels) - 1, -1, -1))
        bars = ax.barh(y_pos, counts, color=bin_colors, edgecolor='black', linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Number of Students', fontsize=10)

        for bar, count in zip(bars, counts):
            pct = count / total * 100
            ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height() / 2,
                    f'{count} ({pct:.1f}%)', va='center', fontsize=9, fontweight='bold')

        title_m = model if model != 'ALLaM-7B' else 'ALLaM-7B (S3)'
        ax.set_title(f'{title_m}\nStudent Difficulty Distribution', fontsize=11, fontweight='bold')

    axes_flat[5].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'figure4_student_difficulty_spectrum.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Done.")


# ============================================================
# Figure 5: Learning Curves (4 models only - copy original)
# ============================================================
def make_fig5():
    src = os.path.join(BASE, "ملف المشروع قبل التعديل", "figure5_temporal_learning_curves.png")
    shutil.copy2(src, os.path.join(OUT, "figure5_temporal_learning_curves.png"))
    print("Figure 5: Copied original (ALLaM excluded - no memory/temporal context).")


# ============================================================
# Figure 6: Error Pattern Analysis (5 models)
# ============================================================
def compute_accuracy_per_question(log_dir):
    """Compute accuracy per question."""
    post_files = glob.glob(os.path.join(log_dir, '*_result_ind_post.csv'))
    q_correct = {}
    q_total = {}
    for f in post_files:
        try:
            df_s = pd.read_csv(f)
            if 'question_id' not in df_s.columns or 'accuracy_similarity' not in df_s.columns:
                continue
            for _, row in df_s.iterrows():
                qid = int(row['question_id'])
                acc = row['accuracy_similarity']
                if pd.isna(acc):
                    continue
                acc = float(acc)
                q_total[qid] = q_total.get(qid, 0) + 1
                q_correct[qid] = q_correct.get(qid, 0) + acc
        except:
            continue
    result = {}
    for qid in sorted(q_total.keys()):
        if q_total[qid] > 0:
            result[qid] = (q_correct[qid] / q_total[qid]) * 100
    return result

def make_fig6():
    print("Generating Figure 6: Error Pattern Analysis (5 models)...")

    sim_folders = {
        'GPT-3.5': 'sim_only_recent_one_reflect-no_KM+PM+MM+CM_standard_cog_example-no_2',
        'GPT-4o Mini': 'sim_only_recent_one_reflect-no_KM+PM+MM+CM_standard_cog_example-no_3',
        'Llama 3.1': 'sim_only_recent_one_reflect-no_KM+PM+MM+CM_standard_cog_example-no_4',
        'Qwen2.5 7B': 'sim_only_recent_one_reflect-no_KM+PM+MM+CM_standard_cog_example-no_5',
        'ALLaM-7B': 'sim_no_memory_reflect-no_KM_standard_example-no_allam_s3',
    }
    real_folders = {
        'GPT-3.5': 'real_only_recent_one_reflect-no_KM+PM+MM+CM_standard_cog_example-no_2',
        'GPT-4o Mini': 'real_only_recent_one_reflect-no_KM+PM+MM+CM_standard_cog_example-no_3',
        'Llama 3.1': 'real_only_recent_one_reflect-no_KM+PM+MM+CM_standard_cog_example-no_4',
        'Qwen2.5 7B': 'real_only_recent_one_reflect-no_KM+PM+MM+CM_standard_cog_example-no_5',
        'ALLaM-7B': 'real_no_memory_reflect-no_KM_standard_example-no_allam_s3',
    }

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle('Error Pattern Analysis Across Questions\n(Gray bars = Question difficulty | Colored line = Model accuracy)',
                 fontsize=14, fontweight='bold')
    axes_flat = axes.flatten()

    for idx, model in enumerate(models_5):
        ax = axes_flat[idx]
        sim_dir = os.path.join(SIM_BASE, sim_folders[model], 'log')
        real_dir = os.path.join(SIM_BASE, real_folders[model], 'log')

        sim_acc = compute_accuracy_per_question(sim_dir)
        real_acc = compute_accuracy_per_question(real_dir)

        if len(sim_acc) == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            continue

        questions = sorted(sim_acc.keys())[:10]
        q_labels = [f'Q{q}' for q in questions]
        real_vals = [real_acc.get(q, 0) for q in questions]
        sim_vals = [sim_acc.get(q, 0) for q in questions]
        x = np.arange(len(questions))

        ax.bar(x, real_vals, color='lightgray', edgecolor='gray', alpha=0.7, label='Question Difficulty')
        color = MODEL_COLORS[model]
        ax.plot(x, sim_vals, 'o-', color=color, linewidth=2, markersize=6,
                label=f'{model} Accuracy')

        ax.set_xticks(x)
        ax.set_xticklabels(q_labels, fontsize=9)
        ax.set_xlabel('Question ID', fontsize=10)
        ax.set_ylabel('Accuracy (%)', fontsize=10)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8, loc='lower left')

        title_m = model if model != 'ALLaM-7B' else 'ALLaM-7B (S3)'
        ax.set_title(f'{title_m}\nError Pattern Analysis', fontsize=11, fontweight='bold')

    axes_flat[5].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'figure6_error_pattern_analysis.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Done.")


# ============================================================
# Figure 7 (NEW): ALLaM Strategy Comparison (S1 vs S2 vs S3)
# ============================================================
def make_fig7():
    print("Generating Figure 7: ALLaM Strategy Comparison...")

    strategies = ['ALLaM S1', 'ALLaM S2', 'ALLaM S3']
    strategy_labels = [
        'S1 (Memory-Free + Cog. Prior)',
        'S2 (LLMLingua-2 Compression)',
        'S3 (Memory-Free, No Cog. Prior)'
    ]
    strategy_colors = ['#3498DB', '#E67E22', '#9B59B6']

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('ALLaM-7B: Prompt Compression Strategy Comparison (Real Students)',
                 fontsize=14, fontweight='bold')

    # Subplot 1: Cognitive State Prediction (MAE)
    ax1 = axes[0]
    cog_labels = ['Focus', 'Engagement', 'Confusion', 'Workload', 'Curiosity']
    cog_keys = ['FOC', 'ENG', 'CO', 'WO', 'CU']
    x = np.arange(len(cog_labels))
    width = 0.25
    for i, skey in enumerate(strategies):
        r = get_allam(skey, 'Real')
        if r is None: continue
        vals = [float(r[k]) for k in cog_keys]
        ax1.bar(x + i * width, vals, width, label=strategy_labels[i],
                color=strategy_colors[i], edgecolor='black', linewidth=0.5)
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(cog_labels, fontsize=10)
    ax1.set_ylabel('MAE (Lower is Better)', fontsize=11)
    ax1.set_title('Cognitive State Prediction', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8)

    # Subplot 2: Assessment Performance
    ax2 = axes[1]
    assess_labels = ['Choice Similarity (%)', 'Correct Answers (%)']
    x2 = np.arange(len(assess_labels))
    for i, skey in enumerate(strategies):
        r = get_allam(skey, 'Real')
        if r is None: continue
        vals = [float(r['CH']), float(r['COR'])]
        bars = ax2.bar(x2 + i * width, vals, width, label=strategy_labels[i],
                       color=strategy_colors[i], edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{val:.1f}', ha='center', fontsize=8, fontweight='bold')
    ax2.set_xticks(x2 + width)
    ax2.set_xticklabels(assess_labels, fontsize=10)
    ax2.set_ylabel('Percentage (Higher is Better)', fontsize=11)
    ax2.set_title('Assessment Performance', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8)

    # Subplot 3: Normalized (higher = better)
    ax3 = axes[2]
    norm_labels = ['Focus\n(inv MAE)', 'Engagement\n(inv MAE)', 'Confusion\n(inv MAE)', 'CH']
    norm_keys = ['FOC', 'ENG', 'CO', 'CH']
    x3 = np.arange(len(norm_labels))
    for i, skey in enumerate(strategies):
        r = get_allam(skey, 'Real')
        if r is None: continue
        vals = []
        for mk in norm_keys:
            v = float(r[mk])
            if mk in ['FOC', 'ENG', 'CO']:
                vals.append(1 - v)
            else:
                vals.append(v / 100.0)
        ax3.bar(x3 + i * width, vals, width, label=strategy_labels[i],
                color=strategy_colors[i], edgecolor='black', linewidth=0.5)
    ax3.set_xticks(x3 + width)
    ax3.set_xticklabels(norm_labels, fontsize=9)
    ax3.set_ylabel('Score (Higher is Better)', fontsize=11)
    ax3.set_title('Normalized Comparison\n(all metrics: higher is better)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'figure7_allam_strategy_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  Done.")


# ============================================================
if __name__ == '__main__':
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    make_fig6()
    make_fig7()
    print(f"\nAll 7 figures saved to: {OUT}")
