# ===========================================
# analysis.py
# Advanced multi-level data analysis + visualization
# ===========================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols, mixedlm
from scipy import stats
import scikit_posthocs as sp
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import plotly.express as px
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings("ignore")

# Raincloud plots (ptitprince)
try:
    import ptitprince as pt
    PTIT_AVAILABLE = True
except Exception:
    PTIT_AVAILABLE = False

# ---------------------------
# Directory setup
# ---------------------------
ROOT = Path.cwd()
DATA_FILE = ROOT / "cycling.txt"
OUT_DIR = ROOT / "output"
PLOTS_DIR = OUT_DIR / "plots"
TABLES_DIR = OUT_DIR / "tables"
LOGS_DIR = OUT_DIR / "logs"

for d in (OUT_DIR, PLOTS_DIR, TABLES_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------
# Load data
# ---------------------------
print("📥 Loading data from:", DATA_FILE)
df = pd.read_csv(DATA_FILE, sep=None, engine="python")
df.columns = [c.strip().replace('"', '') for c in df.columns]
df = df.rename(columns={
    df.columns[0]: "rider",
    df.columns[1]: "rider_class",
    df.columns[2]: "stage",
    df.columns[3]: "points",
    df.columns[4]: "stage_class"
})

df['points'] = pd.to_numeric(df['points'], errors='coerce')
df['rider'] = df['rider'].astype('category')
df['rider_class'] = df['rider_class'].astype('category')
df['stage_class'] = df['stage_class'].astype('category')

print(f"✅ Data shape: {df.shape}")
print(df.head())

# ---------------------------
# Helper functions
# ---------------------------
def save_df(df_obj, path):
    df_obj.to_csv(path, index=True)
    print("💾 Saved:", path)

def eta_squared_anova(aov):
    ss_between = aov['sum_sq'].iloc[0]
    ss_total = aov['sum_sq'].sum()
    return ss_between / ss_total

def epsilon_squared_kw(H, k, N):
    return max(0, (H - k + 1) / (N - k))

# ---------------------------
# Descriptive statistics
# ---------------------------
print("\n📊 Descriptive statistics...")
desc_by_class = df.groupby('rider_class')['points'].agg(
    count='count', mean='mean', median='median', std='std',
    iqr=lambda x: np.subtract(*np.percentile(x.dropna(), [75,25])),
    min='min', max='max'
).round(3)
save_df(desc_by_class, TABLES_DIR / "desc_by_class.csv")

desc_by_stage_class = df.groupby('stage_class')['points'].agg(
    count='count', mean='mean', median='median', std='std',
    iqr=lambda x: np.subtract(*np.percentile(x.dropna(), [75,25])),
    min='min', max='max'
).round(3)
save_df(desc_by_stage_class, TABLES_DIR / "desc_by_stage_class.csv")

desc_stage_and_class = df.groupby(['stage_class','rider_class'])['points'].agg(
    count='count', mean='mean', median='median', std='std',
    iqr=lambda x: np.subtract(*np.percentile(x.dropna(), [75,25]))
).round(3)
save_df(desc_stage_and_class, TABLES_DIR / "desc_by_stage_and_class.csv")

# ---------------------------
# Overall analysis: ANOVA → KW fallback
# ---------------------------
print("\n⚖️ Hypothesis testing (overall)...")
model = ols('points ~ C(rider_class)', data=df).fit()
aov_table = sm.stats.anova_lm(model, typ=2)
save_df(aov_table, TABLES_DIR / "anova_rider_class.csv")

groups = [g['points'].dropna().values for _, g in df.groupby('rider_class')]
levene_stat, levene_p = stats.levene(*groups, center='median')
shap_stat, shap_p = stats.shapiro(model.resid) if len(model.resid)<=5000 else (np.nan, np.nan)
print(f"Levene p={levene_p:.3g}, Shapiro p={shap_p:.3g}")

if (levene_p <= 0.05) or (shap_p <= 0.05):
    print("🚨 Assumptions violated → Using Kruskal–Wallis.")
    H, p_kw = stats.kruskal(*groups, nan_policy='omit')
    eps2 = epsilon_squared_kw(H, df['rider_class'].nunique(), df.shape[0])
    pd.DataFrame([{'H':H,'p':p_kw,'epsilon_sq':eps2}]).to_csv(TABLES_DIR / "kw_overall.csv", index=False)
    print(f"Kruskal-Wallis H={H:.3f}, p={p_kw:.3g}, ε²={eps2:.3f}")

    dunn_overall = sp.posthoc_dunn(df, val_col='points', group_col='rider_class', p_adjust='bonferroni')
    dunn_overall.to_csv(TABLES_DIR / "pairwise_overall_dunn.csv")
else:
    print("✅ ANOVA assumptions OK → Using Tukey HSD.")
    eta2 = eta_squared_anova(aov_table)
    pd.DataFrame([{'eta_squared':eta2}]).to_csv(TABLES_DIR / "eta2_anova_overall.csv", index=False)
    tukey = pairwise_tukeyhsd(endog=df['points'], groups=df['rider_class'], alpha=0.05)
    pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0]).to_csv(TABLES_DIR / "tukey_overall.csv", index=False)

# ---------------------------
# Per-stage analysis
# ---------------------------
print("\n🚴 Per-stage analysis...")
stage_summary = []
for stage in df['stage_class'].cat.categories:
    sub = df[df['stage_class'] == stage]
    groups_stage = [g['points'].dropna().values for _, g in sub.groupby('rider_class')]
    try:
        Hs, ps = stats.kruskal(*groups_stage, nan_policy='omit')
    except Exception:
        Hs, ps = np.nan, np.nan
    eps2_stage = epsilon_squared_kw(Hs, sub['rider_class'].nunique(), sub.shape[0]) if not np.isnan(Hs) else np.nan
    stage_summary.append({'stage':stage,'H':Hs,'p':ps,'epsilon_sq':eps2_stage})
    dunn_stage = sp.posthoc_dunn(sub, val_col='points', group_col='rider_class', p_adjust='bonferroni')
    dunn_stage.to_csv(TABLES_DIR / f"pairwise_{stage}.csv")
    print(f"{stage}: H={Hs:.3f}, p={ps:.3g}, ε²={eps2_stage:.3f}")

stage_summary_df = pd.DataFrame(stage_summary).set_index('stage')
save_df(stage_summary_df, TABLES_DIR / "stagewise_kw_summary.csv")

# ---------------------------
# Mixed-effects model
# ---------------------------
print("\n🧮 Mixed-effects model (random rider intercept)...")
md = mixedlm("points ~ C(rider_class)*C(stage_class)", df, groups=df["rider"])
mdf = md.fit(reml=False)
with open(LOGS_DIR / "mixed_model_summary.txt", "w") as f:
    f.write(mdf.summary().as_text())
print("Mixed model results saved.")

# ---------------------------
# Visualization
# ---------------------------
print("\n🎨 Creating plots...")
sns.set(style="whitegrid", context='notebook')

# Overall violin
plt.figure(figsize=(9,6))
sns.violinplot(data=df, x='rider_class', y='points', inner='quartile', palette='Set2')
plt.title('Overall Distribution of Points by Rider Class')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "violin_overall.png", dpi=300)
plt.close()

# Per-stage violin plots
for stage in df['stage_class'].cat.categories:
    sub = df[df['stage_class'] == stage]
    plt.figure(figsize=(8,5))
    sns.violinplot(data=sub, x='rider_class', y='points', inner='quartile', palette='Set3')
    sns.stripplot(data=sub, x='rider_class', y='points', color='k', size=1.2, alpha=0.3)
    plt.title(f"Points by Rider Class — {stage}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"violin_{stage}.png", dpi=300)
    plt.close()

# Correlation heatmap
print("\n🧩 Creating correlation heatmap...")
corr_data = pd.get_dummies(df[['points', 'rider_class', 'stage_class']], drop_first=True)
corr = corr_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='vlag', center=0)
plt.title("Correlation Heatmap — Points, Rider Class, and Stage Class")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=300)
plt.close()

# ---------------------------
# Dunn Heatmaps (Overall + Per-stage)
# ---------------------------
print("\n🔥 Creating Dunn heatmaps...")
def plot_dunn_heatmap(csv_path, title, save_name):
    dunn = pd.read_csv(csv_path, index_col=0)
    plt.figure(figsize=(6,5))
    sns.heatmap(dunn, annot=True, fmt=".3f", cmap="coolwarm_r", cbar_kws={'label': 'p-value'})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / save_name, dpi=300)
    plt.close()

if (TABLES_DIR / "pairwise_overall_dunn.csv").exists():
    plot_dunn_heatmap(TABLES_DIR / "pairwise_overall_dunn.csv",
                      "Pairwise Dunn Test (Overall, Bonferroni p-values)",
                      "dunn_heatmap_overall.png")

for stage in df['stage_class'].cat.categories:
    path_stage = TABLES_DIR / f"pairwise_{stage}.csv"
    if path_stage.exists():
        plot_dunn_heatmap(path_stage,
                          f"Pairwise Dunn Test — {stage.title()} Stages",
                          f"dunn_heatmap_{stage}.png")

print("✅ Heatmaps saved in plots folder.")

# ---------------------------
# Quick PDF report
# ---------------------------
print("\n🧾 Creating quick PDF report...")
pdf_path = OUT_DIR / "quick_report.pdf"
with PdfPages(pdf_path) as pdf:
    # Page 1 — summary text
    fig_text = plt.figure(figsize=(8.27, 11.69))
    text_lines = [
        "Quick Report",
        "",
        f"Total observations: {df.shape[0]}",
        f"Unique riders: {df['rider'].nunique()}",
        f"Rider classes: {list(df['rider_class'].cat.categories)}",
        f"Stage classes: {list(df['stage_class'].cat.categories)}",
        "",
        "Main Results:",
        f"• Levene p={levene_p:.3g}, Shapiro p={shap_p:.3g}",
        f"• Kruskal–Wallis H={H:.2f}, p={p_kw:.3g}, ε²={eps2:.3f}",
        "",
        "Plots and detailed tables attached in following pages.",
        "",
        "Generated by: analysis.py"
    ]
    fig_text.text(0.05, 0.95, "\n".join(text_lines), va="top", fontsize=10)
    pdf.savefig(fig_text)
    plt.close(fig_text)

    # Adding plots
    for plot_file in ["violin_overall.png", "correlation_heatmap.png", "dunn_heatmap_overall.png"]:
        full_path = PLOTS_DIR / plot_file
        if full_path.exists():
            img = plt.imread(full_path)
            fig = plt.figure(figsize=(8.27, 11.69))
            plt.axis("off")
            plt.imshow(img)
            pdf.savefig(fig)
            plt.close(fig)

print(f"✅ Quick PDF report saved: {pdf_path}")

# ---------------------------
# Quick summary text file
# ---------------------------
summary_lines = [
    f"Total observations: {df.shape[0]}",
    f"Rider Classes: {df['rider_class'].unique().tolist()}",
    f"Stage Classes: {df['stage_class'].unique().tolist()}",
    "\nDescriptive Stats (Mean by Rider Class):"
]
for rc, row in desc_by_class.iterrows():
    summary_lines.append(f"  - {rc}: mean={row['mean']:.2f}, median={row['median']:.2f}, sd={row['std']:.2f}")

summary_lines.append("\nSee tables folder for full stats, effect sizes, and Dunn post-hoc results.")
summary_lines.append("See plots folder for violin, heatmap, and interaction visuals.")
summary_lines.append("Mixed Model Summary: output/logs/mixed_model_summary.txt")

with open(OUT_DIR / "quick_summary.txt", "w") as f:
    f.write("\n".join(summary_lines))

print("\n🎉 ALL DONE — All outputs saved successfully!")
print("Check 'output/' folder for tables, plots, logs, and PDF report.")