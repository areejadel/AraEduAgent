"""
Analyze Results with ALLaM — مقارنة 4 استراتيجيات ALLaM + النماذج الأصلية
==========================================================================
يستخدم النتائج الموجودة فقط — لا يعيد أي تجربة.
"""

import os, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150

BASE_PATH = 'dataset/simulation/full_experiment_300_students'
OUTPUT_DIR = 'dataset/allam_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# تعريف النماذج (الأصلية + ALLaM 4 استراتيجيات)
# ============================================================

MODELS = {
    'GPT-3.5':              {'suffix': '2'},
    'GPT-4o Mini':          {'suffix': '3'},
    'Llama 3.1':            {'suffix': '4'},
    'Qwen2.5 7B':           {'suffix': '5'},
    'ALLaM-S0 (Original)':  {'suffix': 'allam_s0'},
    'ALLaM-S1 (NoMemory)':  {'suffix': 'allam_s1'},
    'ALLaM-S2 (Compressed)':{'suffix': 'allam_s2'},
    'ALLaM-S3 (Chunked)':   {'suffix': 'allam_s3'},
}

DUR_HEADER = ['reflection_choice','memory_component_choice','memory_source','forget_effect',
              'sim_strategy','example_demo','gpt_type','student_id','transcript_id','sentence_id',
              'user_gaze_aoi_id','user_gaze_aoi_center_tuple_x','user_gaze_aoi_center_tuple_y',
              'user_motor_aoi_id','user_motor_aoi_center_tuple_x','user_motor_aoi_center_tuple_y',
              'user_workload','user_curiosity','user_valid_focus','user_course_follow',
              'user_engagement','user_confusion',
              'agent_gaze_aoi_id','agent_gaze_aoi_center_tuple_x','agent_gaze_aoi_center_tuple_y',
              'agent_motor_aoi_id','agent_motor_aoi_center_tuple_x','agent_motor_aoi_center_tuple_y',
              'agent_workload','agent_curiosity','agent_valid_focus','agent_course_follow',
              'agent_engagement','agent_confusion',
              'gaze_aoi_accuracy','gaze_aoi_distance','motor_aoi_accuracy','motor_aoi_distance',
              'workload_diff','curiosity_diff','valid_focus_diff','follow_ratio_diff',
              'engagement_accuracy','confusion_accuracy']

POST_HEADER = ['reflection_choice','memory_component_choice','memory_source','forget_effect',
               'sim_strategy','example_demo','gpt_type','student_id','question_id',
               'user_answer','agent_answer','correct_answer','choice_similarity','accuracy_similarity']


def load_results(model_name, source='real'):
    """Load dur and post CSVs for a model."""
    suffix = MODELS[model_name]['suffix']
    pattern = f"{source}_*_{suffix}"
    # Find matching folder
    matching = [d for d in os.listdir(BASE_PATH) if d.endswith(f'_{suffix}') and d.startswith(source)]
    if not matching:
        return None, None
    folder = matching[0]
    log_dir = os.path.join(BASE_PATH, folder, 'log')
    if not os.path.exists(log_dir):
        return None, None

    dur_files = glob.glob(os.path.join(log_dir, '*_result_ind_dur.csv'))
    post_files = glob.glob(os.path.join(log_dir, '*_result_ind_post.csv'))

    dur_dfs = []
    for f in dur_files:
        try:
            df = pd.read_csv(f, encoding='utf-8-sig')
            dur_dfs.append(df)
        except:
            pass
    dur = pd.concat(dur_dfs, ignore_index=True) if dur_dfs else None

    post_dfs = []
    for f in post_files:
        try:
            df = pd.read_csv(f, encoding='utf-8-sig')
            post_dfs.append(df)
        except:
            pass
    post = pd.concat(post_dfs, ignore_index=True) if post_dfs else None

    return dur, post


def compute_metrics(dur, post):
    """Compute standard benchmark metrics."""
    m = {}
    if dur is not None and len(dur) > 0:
        for col, key in [('gaze_aoi_distance','GA↓'), ('motor_aoi_distance','MO↓'),
                         ('workload_diff','WO↓'), ('curiosity_diff','CU↓'),
                         ('valid_focus_diff','FOC↓'), ('follow_ratio_diff','FOL↓'),
                         ('engagement_accuracy','ENG↓'), ('confusion_accuracy','CO↓')]:
            if col in dur.columns:
                v = pd.to_numeric(dur[col], errors='coerce').dropna()
                m[key] = round(v.mean(), 3) if len(v) > 0 else None
        m['N'] = dur['student_id'].nunique() if 'student_id' in dur.columns else 0
    if post is not None and len(post) > 0:
        if 'choice_similarity' in post.columns:
            v = pd.to_numeric(post['choice_similarity'], errors='coerce').dropna()
            m['CH↑'] = round(v.mean() * 100, 1) if len(v) > 0 else None
        if 'accuracy_similarity' in post.columns:
            v = pd.to_numeric(post['accuracy_similarity'], errors='coerce').dropna()
            m['AC↑'] = round(v.mean() * 100, 1) if len(v) > 0 else None
    return m


def build_benchmark_table():
    """Build complete benchmark table."""
    print("\n📊 Building Benchmark Table...")
    rows = []
    for model_name in MODELS:
        for source in ['real', 'sim']:
            dur, post = load_results(model_name, source)
            if dur is None and post is None:
                print(f"  ⚠️  {model_name} ({source}): No data found")
                continue
            metrics = compute_metrics(dur, post)
            n = metrics.pop('N', 0)
            if n == 0 and dur is None:
                continue
            row = {'Model': model_name, 'Exp': source.upper(), 'N': n}
            row.update(metrics)
            rows.append(row)
            print(f"  ✅ {model_name} ({source}): N={n}")

    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, 'benchmark_table.csv')
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f"\n💾 Saved: {path}")
    if len(df) > 0:
        print(df.to_string(index=False))
    return df


def plot_allam_strategy_comparison(df):
    """Compare 4 ALLaM strategies."""
    allam = df[df['Model'].str.contains('ALLaM', na=False)]
    real_allam = allam[allam['Exp'] == 'REAL']
    if len(real_allam) == 0:
        print("  ⚠️  No ALLaM Real data for strategy comparison")
        return

    cog_cols = ['WO↓', 'CU↓', 'FOC↓', 'FOL↓', 'ENG↓', 'CO↓']
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 1. Cognitive MAE per strategy
    ax = axes[0]
    models = real_allam['Model'].values
    x = np.arange(len(cog_cols))
    width = 0.8 / len(models)
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    for i, (_, row) in enumerate(real_allam.iterrows()):
        vals = [row.get(c, 0) if pd.notna(row.get(c, None)) else 0 for c in cog_cols]
        ax.bar(x + i * width, vals, width, label=row['Model'], color=colors[i % len(colors)], alpha=0.85)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([c.replace('↓','') for c in cog_cols])
    ax.set_ylabel('MAE (lower = better)')
    ax.set_title('Cognitive Metrics by Strategy')
    ax.legend(fontsize=7, loc='upper right')

    # 2. Behavioral distances
    ax = axes[1]
    ga_vals = real_allam['GA↓'].fillna(0).values
    mo_vals = real_allam['MO↓'].fillna(0).values
    x2 = np.arange(len(models))
    ax.bar(x2 - 0.15, ga_vals, 0.3, label='Gaze Distance', color='#4ECDC4')
    ax.bar(x2 + 0.15, mo_vals, 0.3, label='Motor Distance', color='#FF6B6B')
    ax.set_xticks(x2)
    ax.set_xticklabels([m.split('(')[1].rstrip(')') for m in models], fontsize=9)
    ax.set_ylabel('Distance (lower = better)')
    ax.set_title('Behavioral Metrics by Strategy')
    ax.legend()

    # 3. Assessment accuracy
    ax = axes[2]
    ch_vals = real_allam['CH↑'].fillna(0).values
    ac_vals = real_allam['AC↑'].fillna(0).values
    ax.bar(x2 - 0.15, ch_vals, 0.3, label='Choice Similarity %', color='#9B59B6')
    ax.bar(x2 + 0.15, ac_vals, 0.3, label='Accuracy Similarity %', color='#F39C12')
    ax.set_xticks(x2)
    ax.set_xticklabels([m.split('(')[1].rstrip(')') for m in models], fontsize=9)
    ax.set_ylabel('Percentage (higher = better)')
    ax.set_title('Assessment Metrics by Strategy')
    ax.legend()

    plt.suptitle('ALLaM: Comparison of 4 Context-Reduction Strategies', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'allam_4_strategies_comparison.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved: {path}")


def plot_all_models_comparison(df):
    """Compare ALL models (original 4 + best ALLaM) for paper."""
    real_df = df[df['Exp'] == 'REAL'].copy()
    if len(real_df) == 0:
        print("  ⚠️  No Real data")
        return

    cog_cols = ['WO↓', 'CU↓', 'FOC↓', 'FOL↓', 'ENG↓', 'CO↓']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Cognitive heatmap
    ax = axes[0][0]
    heatmap_data = real_df.set_index('Model')[cog_cols].astype(float)
    if len(heatmap_data) > 0:
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=ax, cbar_kws={'label': 'MAE'})
        ax.set_title('Cognitive State MAE (lower = better)')

    # 2. Gaze + Motor bar
    ax = axes[0][1]
    x = np.arange(len(real_df))
    ax.bar(x - 0.15, real_df['GA↓'].fillna(0), 0.3, label='Gaze Distance', color='#4ECDC4')
    ax.bar(x + 0.15, real_df['MO↓'].fillna(0), 0.3, label='Motor Distance', color='#FF6B6B')
    ax.set_xticks(x)
    ax.set_xticklabels(real_df['Model'], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Distance')
    ax.set_title('Behavioral Distance (lower = better)')
    ax.legend()

    # 3. Assessment
    ax = axes[1][0]
    ax.bar(x - 0.15, real_df['CH↑'].fillna(0), 0.3, label='Choice Sim %', color='#9B59B6')
    ax.bar(x + 0.15, real_df['AC↑'].fillna(0), 0.3, label='Accuracy Sim %', color='#F39C12')
    ax.set_xticks(x)
    ax.set_xticklabels(real_df['Model'], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Percentage')
    ax.set_title('Assessment (higher = better)')
    ax.legend()

    # 4. Overall rank
    ax = axes[1][1]
    rank_cols = ['GA↓', 'MO↓'] + cog_cols
    available_cols = [c for c in rank_cols if c in real_df.columns]
    if available_cols:
        avg_mae = real_df[available_cols].mean(axis=1).fillna(999)
        colors = ['#2ecc71' if 'ALLaM' in m else '#3498db' for m in real_df['Model']]
        bars = ax.barh(real_df['Model'], avg_mae, color=colors)
        ax.set_xlabel('Average MAE (lower = better)')
        ax.set_title('Overall Score (distance + cognitive)')
        ax.invert_yaxis()

    plt.suptitle('AraEduAgent: All Models Benchmark (Real Condition)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'all_models_benchmark.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved: {path}")


def main():
    print("=" * 70)
    print("📊 AraEduAgent Analysis — Including ALLaM Strategies")
    print("=" * 70)

    df = build_benchmark_table()
    if len(df) == 0:
        print("\n❌ No results found. Run experiments first!")
        return

    print("\n📈 Creating visualizations...")
    plot_allam_strategy_comparison(df)
    plot_all_models_comparison(df)

    # LaTeX table
    latex_path = os.path.join(OUTPUT_DIR, 'benchmark_table.tex')
    df.to_latex(latex_path, index=False, float_format="%.3f")
    print(f"💾 LaTeX table: {latex_path}")

    print(f"\n✅ All results saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
