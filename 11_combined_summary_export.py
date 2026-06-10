# ============================================================
# 11_combined_summary_export.py
# HGT Spatial Dependence — GBD East Africa
# Combined Summary, Cross-Graph Comparison & Export
#
# Part of: HGT_GBD-Mortality
# Repository: https://github.com/SallySims/HGT_GBD-Mortality
#
# Run order: 11 of 11
# Prerequisites: Run files 01 through 10 first
# ============================================================

# ============================================================
# CELL 11: COMBINED SUMMARY, CROSS-GRAPH COMPARISON & EXPORT
# ============================================================

all_results = pd.concat(
    [pd.DataFrame(results_A), pd.DataFrame(results_B)],
    ignore_index=True)

summary_cols = [
    'Graph','Model','Disease',
    'R2_Base_Mean','R2_Base_SD',
    'R2_Spatial_Mean','R2_Spatial_SD',
    'Improvement_%','CI_Lo_%','CI_Hi_%',
    'Morans_I','Morans_p',
    'OLS_SpatLag_R2','CAR_R2',
    'Label'
]

print("\n" + "="*110)
print("FULL RESULTS — GRAPH A (80 nodes) vs GRAPH B (160 nodes)")
print("="*110)
print(all_results[summary_cols].to_string(index=False))

# Cross-graph pivot
print("\n" + "─"*80)
print("CROSS-GRAPH: Mean Improvement % by Model × Disease")
pivot = all_results.pivot_table(
    index=['Model','Disease'],
    columns='Graph',
    values='Improvement_%').round(2)
print(pivot.to_string())

# Benchmark comparison
print("\n" + "─"*80)
print("BENCHMARK COMPARISON: HGT vs OLS Spatial Lag vs CAR")
bench_cols=['Graph','Model','Disease',
            'R2_Spatial_Mean','OLS_SpatLag_R2','CAR_R2',
            'MSE_Spatial','OLS_SpatLag_MSE','CAR_MSE']
print(all_results[bench_cols].to_string(index=False))

# Export
all_results.to_csv('HGT_Results_Full.csv', index=False)
rolling_df.to_csv('HGT_RollingWindows.csv', index=False)
ablation_df.to_csv('HGT_Ablation.csv', index=False)
mc_df.to_csv('HGT_MC_Uncertainty.csv', index=False)
print("\nSaved: HGT_Results_Full.csv, HGT_RollingWindows.csv,")
print("       HGT_Ablation.csv, HGT_MC_Uncertainty.csv")

print("""
NOTE ON INTERPRETATION
──────────────────────────────────────────────────────────────────────
Improvement % = relative MSE reduction from adding spatial edges.
Reflects PREDICTIVE GAINS from spatial graph structure only.
NOT formal proof of causal cross-border spillovers.

Moran's I: formal test of residual spatial autocorrelation.
Significant (p<0.05) = unexplained spatial structure remains.

OLS Spatial Lag / CAR: traditional spatial benchmarks.
HGT gains over these reflect value of non-linear heterogeneous modelling.

Graph A vs Graph B differences: indicate sex-specific spatial
heterogeneity in NCD mortality patterns.

MC uncertainty: quantifies how much result variability stems from
GBD measurement uncertainty vs model randomness.
──────────────────────────────────────────────────────────────────────
""")
