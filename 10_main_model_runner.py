# ============================================================
# 10_main_model_runner.py
# HGT Spatial Dependence — GBD East Africa
# Main Model Runner (All Models × Both Graphs)
#
# Part of: HGT_GBD-Mortality
# Repository: https://github.com/SallySims/HGT_GBD-Mortality
#
# Run order: 10 of 11
# Prerequisites: Run files 01 through 09 first
# ============================================================

# ============================================================
# CELL 10: MAIN MODEL RUNNER — ALL MODELS × BOTH GRAPHS
# ============================================================

def run_all_models(df_in, graph_tag, build_fn, node_key_fn):
    """
    Run SDI-only, Risk-only, and Full models for all diseases.
    Prints graph structure summary on first run.
    Returns list of result dicts.
    """
    all_results = []
    printed_summary = False

    for model_name, use_sdi, use_risk in [
        ("SDI Only",  True,  False),
        ("Risk Only", False, True ),
        ("SDI+Risk",  True,  True ),
    ]:
        print(f"\n{'─'*65}")
        print(f"[{graph_tag}] {model_name}")
        print(f"{'─'*65}")

        if use_risk:
            df_m = df_in[df_in['risk_factor'].isin(RISK_LIST)].copy()
        else:
            df_m = df_in.copy()

        for disease in sorted(df_m['cause'].unique()):
            df_d = df_m[df_m['cause']==disease].copy()
            train_df = df_d[df_d['year']<=TRAIN_YEAR_CUTOFF].copy()
            test_df  = df_d[df_d['year'] >TRAIN_YEAR_CUTOFF].copy()

            t_sc=StandardScaler()
            train_df['y']=t_sc.fit_transform(train_df[['deathratevalue']])
            test_df['y'] =t_sc.transform(test_df[['deathratevalue']])
            y_sc=StandardScaler(); y_sc.fit(train_df[['year']])
            train_df['year_s']=y_sc.transform(train_df[['year']])
            test_df['year_s'] =y_sc.transform(test_df[['year']])

            if use_sdi:
                train_df['sdi_s'],sdi_sc=encode_sdi_continuous(
                    train_df['SDI_Quintile'],fit=True)
                test_df['sdi_s']=encode_sdi_continuous(
                    test_df['SDI_Quintile'],scaler=sdi_sc)
            if use_risk:
                for r in RISK_LIST:
                    train_df[f'risk_{r}']=(train_df['risk_factor']==r).astype(float)
                    test_df[f'risk_{r}'] =(test_df['risk_factor'] ==r).astype(float)

            le=LabelEncoder()
            ak=sorted(set(train_df.apply(node_key_fn,axis=1))
                    | set(test_df.apply(node_key_fn,axis=1)))
            le.fit(ak)

            br=run_multiseed(
                lambda: build_fn(train_df,le,spatial=False),
                lambda: build_fn(test_df, le,spatial=False), t_sc)
            sr=run_multiseed(
                lambda: build_fn(train_df,le,spatial=True),
                lambda: build_fn(test_df, le,spatial=True),  t_sc)

            # Print graph summary once
            if not printed_summary:
                g = build_fn(train_df, le, spatial=True)
                print_graph_summary(g, f"{graph_tag} [{model_name}]")
                printed_summary = True

            impr=((br['mse_mean']-sr['mse_mean'])/br['mse_mean']*100
                  if br['mse_mean']>0 else 0.0)
            ci_lo,ci_hi=boot_ci(br['mse_mean'],br['mse_sd'],
                                sr['mse_mean'],sr['mse_sd'])
            mI,mp=morans_i(sr['residuals'],test_df)

            bench_feat=['age_mid','year']
            if use_sdi:  bench_feat.append('sdi_value')
            if use_risk: bench_feat+=[f'risk_{r}' for r in RISK_LIST
                                      if f'risk_{r}' in train_df.columns]
            ols_mse,ols_r2 = spatial_lag_ols(train_df,test_df,
                                ['age_mid','year'])
            car_mse,car_r2 = car_benchmark(train_df,test_df,
                                ['age_mid','year'])

            label=("Strong predictive gain"   if impr>8 else
                   "Moderate predictive gain" if impr>3 else
                   "Weak / no predictive gain")
            diag=("" if impr>=0 else
                  f"Spatial edges degrade predictions — no genuine spatial signal "
                  f"detected; adjacency introduces noise into attention mechanism.")

            print(f"  {disease:30s} | "
                  f"Base R²={br['r2_mean']:.3f}±{br['r2_sd']:.3f} "
                  f"Spat R²={sr['r2_mean']:.3f}±{sr['r2_sd']:.3f} "
                  f"Impr={impr:.1f}% [{ci_lo},{ci_hi}] "
                  f"Moran I={mI}(p={mp})")
            if diag: print(f"    ⚠ {diag}")

            all_results.append({
                'Graph':graph_tag,'Model':model_name,'Disease':disease,
                'TrainLoss_Base':round(br['loss_mean'],4),
                'TrainLoss_Spatial':round(sr['loss_mean'],4),
                'R2_Base_Mean':round(br['r2_mean'],3),
                'R2_Base_SD':round(br['r2_sd'],3),
                'R2_Spatial_Mean':round(sr['r2_mean'],3),
                'R2_Spatial_SD':round(sr['r2_sd'],3),
                'MSE_Base':round(br['mse_mean'],4),
                'MSE_Spatial':round(sr['mse_mean'],4),
                'Improvement_%':round(impr,2),
                'CI_Lo_%':ci_lo,'CI_Hi_%':ci_hi,
                'Morans_I':mI,'Morans_p':mp,
                'OLS_SpatLag_R2':ols_r2,'OLS_SpatLag_MSE':ols_mse,
                'CAR_R2':car_r2,'CAR_MSE':car_mse,
                'Label':label,'Diagnostic':diag
            })
    return all_results


# ── Graph A ──
print("\n" + "="*70)
print("GRAPH A: Both-sex × Age × Country  (80 nodes)")
print("="*70)
dfA_main = dfA[['location','age','cause','year','deathratevalue',
                'SDI_Quintile','risk_factor','age_mid',
                'upper','lower']].dropna(
    subset=['location','age','cause','year','deathratevalue']).copy()

results_A = run_all_models(
    dfA_main, "A",
    build_graph_A,
    lambda r: f"{r['location']}|{r['age']}")

# ── Graph B ──
print("\n" + "="*70)
print("GRAPH B: Male/Female × Age × Country  (160 nodes)")
print("="*70)
dfB_main = dfB[['location','sex','age','cause','year','deathratevalue',
                'SDI_Quintile','risk_factor','age_mid',
                'upper','lower']].dropna(
    subset=['location','sex','age','cause','year','deathratevalue']).copy()

results_B = run_all_models(
    dfB_main, "B",
    build_graph_B,
    lambda r: f"{r['location']}|{r['sex']}|{r['age']}")
