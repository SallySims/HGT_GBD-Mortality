# ============================================================
# 07_rolling_temporal_windows.py
# HGT Spatial Dependence — GBD East Africa
# Rolling Temporal Window Validation
#
# Part of: HGT_GBD-Mortality
# Repository: https://github.com/SallySims/HGT_GBD-Mortality
#
# Run order: 07 of 11
# Prerequisites: Run files 01 through 06 first
# ============================================================

# ============================================================
# CELL 7: ROLLING TEMPORAL WINDOW VALIDATION
# Addresses static adjacency concern (Reviewer 1)
# ============================================================

def run_rolling_windows(df_in, disease, build_fn, le_node,
                        node_key_fn, use_sdi, use_risk, tag):
    """
    Four expanding training windows, each tested on the next period.
    Shows whether spatial dependence patterns are stable or shift over time.
    Addresses reviewer concern that a static adjacency graph across 33 years
    may oversimplify evolving regional relationships.
    """
    results = []
    for (tr_start,tr_end,te_start,te_end,wname) in ROLLING_WINDOWS:
        df_d = df_in[df_in['cause']==disease].copy()
        train_df = df_d[(df_d['year']>=tr_start)&(df_d['year']<=tr_end)].copy()
        test_df  = df_d[(df_d['year']>=te_start)&(df_d['year']<=te_end)].copy()

        if len(train_df)==0 or len(test_df)==0:
            continue

        t_sc=StandardScaler()
        train_df['y'] = t_sc.fit_transform(train_df[['deathratevalue']])
        test_df['y']  = t_sc.transform(test_df[['deathratevalue']])

        y_sc=StandardScaler(); y_sc.fit(train_df[['year']])
        train_df['year_s']=y_sc.transform(train_df[['year']])
        test_df['year_s'] =y_sc.transform(test_df[['year']])

        if use_sdi:
            train_df['sdi_s'],sdi_sc=encode_sdi_continuous(
                train_df['SDI_Quintile'],fit=True)
            test_df['sdi_s']=encode_sdi_continuous(
                test_df['SDI_Quintile'],scaler=sdi_sc)
        if use_risk and 'risk_factor' in df_d.columns:
            for r in RISK_LIST:
                train_df[f'risk_{r}']=(train_df['risk_factor']==r).astype(float)
                test_df[f'risk_{r}'] =(test_df['risk_factor'] ==r).astype(float)

        le=LabelEncoder()
        ak=sorted(set(train_df.apply(node_key_fn,axis=1))
                | set(test_df.apply(node_key_fn,axis=1)))
        le.fit(ak)

        br=run_multiseed(
            lambda: build_fn(train_df, le, spatial=False),
            lambda: build_fn(test_df,  le, spatial=False), t_sc)
        sr=run_multiseed(
            lambda: build_fn(train_df, le, spatial=True),
            lambda: build_fn(test_df,  le, spatial=True),  t_sc)

        impr=((br['mse_mean']-sr['mse_mean'])/br['mse_mean']*100
              if br['mse_mean']>0 else 0.0)

        print(f"  {disease} [{tag}] {wname}: "
              f"Base R²={br['r2_mean']:.3f} Spat R²={sr['r2_mean']:.3f} "
              f"Impr={impr:.1f}%")
        results.append({'Disease':disease,'Window':wname,
                        'Train':f"{tr_start}-{tr_end}",
                        'Test':f"{te_start}-{te_end}",
                        'R2_Base':round(br['r2_mean'],3),
                        'R2_Spatial':round(sr['r2_mean'],3),
                        'Improvement_%':round(impr,2)})
    return results

print("Running rolling temporal window validation (Full model, Graph A)...")
rolling_rows = []

dfA_full_roll = dfA[dfA['risk_factor'].isin(RISK_LIST)][
    ['location','age','cause','year','deathratevalue',
     'SDI_Quintile','risk_factor','age_mid']].dropna().copy()

for disease in sorted(dfA_full_roll['cause'].unique()):
    rows = run_rolling_windows(
        dfA_full_roll, disease, build_graph_A, LabelEncoder(),
        lambda r: f"{r['location']}|{r['age']}",
        use_sdi=True, use_risk=True, tag="GraphA-Full")
    rolling_rows.extend(rows)

rolling_df = pd.DataFrame(rolling_rows)
print("\nRolling Window Summary:")
print(rolling_df.to_string(index=False))

# Plot improvement over time per disease
diseases_plot = rolling_df['Disease'].unique()
fig, axes = plt.subplots(1, len(diseases_plot),
                         figsize=(4*len(diseases_plot), 4), sharey=False)
if len(diseases_plot)==1: axes=[axes]
for ax, dis in zip(axes, diseases_plot):
    sub = rolling_df[rolling_df['Disease']==dis]
    ax.plot(sub['Window'], sub['Improvement_%'], marker='o', color='steelblue')
    ax.axhline(0, color='red', linestyle='--', linewidth=0.8)
    ax.set_title(dis); ax.set_xlabel('Window')
    ax.set_ylabel('Improvement %'); ax.tick_params(axis='x', rotation=30)
plt.suptitle("Spatial Improvement Stability Across Time Windows", fontsize=12)
plt.tight_layout()
plt.savefig('rolling_window_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: rolling_window_results.png")
