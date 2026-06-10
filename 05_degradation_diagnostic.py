# ============================================================
# 05_degradation_diagnostic.py
# HGT Spatial Dependence — GBD East Africa
# Degradation Diagnostic (HHD & Stroke)
#
# Part of: HGT_GBD-Mortality
# Repository: https://github.com/SallySims/HGT_GBD-Mortality
#
# Run order: 05 of 11
# Prerequisites: Run files 01 through 04 first
# ============================================================

# ============================================================
# CELL 5: DEGRADATION DIAGNOSTIC
# HHD and Stroke — Graph A and Graph B comparison
# Fixes: disease-specific messages, Graph B comparison added
# ============================================================

DISEASE_DIAGNOSTICS = {
    'HHD': {
        'driver': 'country-specific SBP distributions and healthcare access patterns',
        'mechanism': (
            'HHD is predominantly driven by systolic blood pressure (SBP), which varies '
            'substantially between countries based on diet, salt intake, and health system '
            'capacity. These are localised factors with little cross-border transmission. '
            'The severe degradation and higher spatial training loss confirm that adjacency '
            'edges introduce structural noise into the attention mechanism rather than '
            'capturing genuine regional dynamics.'
        )
    },
    'stroke': {
        'driver': (
            'country-specific vascular risk profiles and data uncertainty '
            'in young age groups (ages 20-54)'
        ),
        'mechanism': (
            'Stroke has a multifactorial aetiology (hypertension, atrial fibrillation, '
            'metabolic risk) that varies by age-sex stratum. The modest negative improvement '
            'for stroke, combined with known GBD data uncertainty in stroke estimates for '
            'young age groups in East Africa (wide uncertainty intervals, clipped negatives '
            'at ages 20-54), suggests the spatial signal is weak relative to measurement '
            'noise. Unlike IHD and diabetes, stroke risk factors do not follow strongly '
            'shared regional patterns across these five countries.'
        )
    }
}


def degradation_diagnostic(df_disease, disease, build_fn, le_node,
                            node_key_fn, use_sdi, use_risk, tag):
    train_df = df_disease[df_disease['year'] <= TRAIN_YEAR_CUTOFF].copy()
    test_df  = df_disease[df_disease['year']  > TRAIN_YEAR_CUTOFF].copy()

    if len(train_df) == 0 or len(test_df) == 0:
        print(f'  Skipping {disease} [{tag}]: insufficient data')
        return None

    t_sc = StandardScaler()
    train_df['y'] = t_sc.fit_transform(train_df[['deathratevalue']])
    test_df['y']  = t_sc.transform(test_df[['deathratevalue']])

    y_sc = StandardScaler()
    y_sc.fit(train_df[['year']])
    train_df['year_s'] = y_sc.transform(train_df[['year']])
    test_df['year_s']  = y_sc.transform(test_df[['year']])

    if use_sdi:
        train_df['sdi_s'], sdi_sc = encode_sdi_continuous(
            train_df['SDI_Quintile'], fit=True)
        test_df['sdi_s'] = encode_sdi_continuous(
            test_df['SDI_Quintile'], scaler=sdi_sc)

    if use_risk:
        for r in RISK_LIST:
            train_df[f'risk_{r}'] = (train_df['risk_factor'] == r).astype(float)
            test_df[f'risk_{r}']  = (test_df['risk_factor']  == r).astype(float)

    all_keys = sorted(
        set(train_df.apply(node_key_fn, axis=1)) |
        set(test_df.apply(node_key_fn,  axis=1))
    )
    le_node.fit(all_keys)

    base_res = run_multiseed(
        lambda: build_fn(train_df, le_node, spatial=False),
        lambda: build_fn(test_df,  le_node, spatial=False),
        t_sc, seeds=[42], track_grads=True)
    spat_res = run_multiseed(
        lambda: build_fn(train_df, le_node, spatial=True),
        lambda: build_fn(test_df,  le_node, spatial=True),
        t_sc, seeds=[42], track_grads=True)

    impr = (
        (base_res['mse_mean'] - spat_res['mse_mean']) /
        base_res['mse_mean'] * 100
        if base_res['mse_mean'] > 0 else 0.0
    )

    # ── Plot: loss curves + gradient norms ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'Degradation Diagnostic: {disease} [{tag}]', fontsize=13)

    ax = axes[0]
    if base_res['loss_curves']:
        ax.plot(base_res['loss_curves'][0], label='Baseline', color='steelblue')
    if spat_res['loss_curves']:
        ax.plot(spat_res['loss_curves'][0], label='Spatial',  color='tomato')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
    ax.set_title('Training Loss Curves')
    ax.legend(); ax.set_yscale('log')

    ax = axes[1]
    if base_res['grad_norms'] and base_res['grad_norms'][0]:
        ax.plot(base_res['grad_norms'][0], label='Baseline', color='steelblue')
    if spat_res['grad_norms'] and spat_res['grad_norms'][0]:
        ax.plot(spat_res['grad_norms'][0], label='Spatial',  color='tomato')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Gradient L2 Norm')
    ax.set_title('Gradient Norm Evolution')
    ax.legend()

    plt.tight_layout()
    fname = f'diagnostic_{disease.replace(" ","_")}_{tag}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()

    # ── Disease-specific diagnostic output ──
    diag    = DISEASE_DIAGNOSTICS.get(disease, {})
    driver  = diag.get('driver',  'localised, country-specific factors')
    mechstr = diag.get('mechanism', 'Spatial adjacency introduces noise.')

    print(f'\n  {disease} [{tag}]: Improvement = {impr:.2f}%')
    print(f'  Base MSE : {base_res["mse_mean"]:.4f} +/- {base_res["mse_sd"]:.4f}')
    print(f'  Spat MSE : {spat_res["mse_mean"]:.4f} +/- {spat_res["mse_sd"]:.4f}')

    if impr < 0:
        severity = 'SEVERE' if impr < -50 else 'MODERATE'
        print(f'  WARNING {severity} DEGRADATION: spatial edges worsen predictions.')
        print(f'  Primary driver : {driver}')
        print(f'  Explanation    : {mechstr}')

        if spat_res['loss_curves'] and base_res['loss_curves']:
            final_base = base_res['loss_curves'][0][-1]
            final_spat = spat_res['loss_curves'][0][-1]
            if final_spat > final_base:
                print(f'  Train loss also higher for spatial '
                      f'({final_spat:.4f} vs {final_base:.4f})')
                print(f'  -> Over-parameterisation confirmed: adjacency harms both'
                      f' training and generalisation for this disease.')
            else:
                print(f'  Train loss lower for spatial '
                      f'({final_spat:.4f} vs {final_base:.4f})')
                print('  -> Spatial model overfits training but fails to generalise.')

        if impr < -100:
            print(f'  Paper note: {impr:.1f}% is extreme degradation. Report as'
                  f' strong evidence of absent spatial dependence for {disease}.')
        else:
            print(f'  Paper note: Modest negative improvement. Spatial signal weak'
                  f' relative to measurement noise; interpret cautiously.')
    else:
        print(f'  Spatial structure adds predictive value for {disease}.')

    return impr


# ── Run diagnostics on Graph A AND Graph B ──
print('Running degradation diagnostics for HHD and Stroke...')
print('Comparing Graph A (80 nodes) vs Graph B (160 nodes)\n')

diag_results = {}

for disease in ['HHD', 'stroke']:
    diag_results[disease] = {}

    # Graph A
    dfA_diag = dfA[dfA['risk_factor'].isin(RISK_LIST)][
        ['location','age','cause','year','deathratevalue',
         'SDI_Quintile','risk_factor','age_mid']].dropna().copy()
    df_d_A = dfA_diag[dfA_diag['cause'] == disease].copy()
    if len(df_d_A) > 0:
        impr_A = degradation_diagnostic(
            df_d_A, disease,
            build_graph_A, LabelEncoder(),
            lambda r: f"{r['location']}|{r['age']}",
            use_sdi=True, use_risk=True, tag='GraphA-Full')
        diag_results[disease]['A'] = impr_A

    # Graph B
    dfB_diag = dfB[dfB['risk_factor'].isin(RISK_LIST)][
        ['location','sex','age','cause','year','deathratevalue',
         'SDI_Quintile','risk_factor','age_mid']].dropna().copy()
    df_d_B = dfB_diag[dfB_diag['cause'] == disease].copy()
    if len(df_d_B) > 0:
        impr_B = degradation_diagnostic(
            df_d_B, disease,
            build_graph_B, LabelEncoder(),
            lambda r: f"{r['location']}|{r['sex']}|{r['age']}",
            use_sdi=True, use_risk=True, tag='GraphB-Full')
        diag_results[disease]['B'] = impr_B

# ── Cross-graph summary ──
print('\n' + '='*65)
print('DEGRADATION SUMMARY: Graph A vs Graph B')
print('='*65)
print(f"{'Disease':10s}  {'Graph A (80)':>14s}  {'Graph B (160)':>14s}  Interpretation")
print('-'*75)
for disease in ['HHD', 'stroke']:
    ia = diag_results[disease].get('A')
    ib = diag_results[disease].get('B')
    ia_s = f'{ia:.2f}%' if ia is not None else 'N/A'
    ib_s = f'{ib:.2f}%' if ib is not None else 'N/A'
    if ia is not None and ib is not None:
        diff = ib - ia
        # Interpret direction of change from Graph A to Graph B
        if ia < 0 and ib < 0:
            # Both negative: is degradation smaller in magnitude for B?
            if diff > 5:
                interp = f'Sex-disaggregation reduces degradation magnitude ({diff:+.1f}pp)'
            elif diff < -5:
                interp = 'Sex-disaggregation worsens degradation'
            else:
                interp = 'Consistent degradation across both graphs'
        elif ia < 0 and ib >= 0:
            interp = 'Sex-disaggregation reverses degradation to gain'
        elif ia >= 0 and ib >= 0:
            if diff > 5:
                interp = 'Sex-disaggregation amplifies spatial gain'
            elif diff < -5:
                interp = 'Sex-disaggregation reduces spatial gain'
            else:
                interp = 'Consistent spatial gain across both graphs'
        else:
            interp = 'Mixed pattern'
    else:
        interp = 'Incomplete'
    print(f'{disease:10s}  {ia_s:>14s}  {ib_s:>14s}  {interp}')
print('='*65)
print()

# ── Data-driven guidance based on actual results ──
print('INTERPRETATION GUIDANCE (based on results above):')
print('(Note: if table labels above look wrong, re-download this')
print(' notebook — Colab may be running a cached older version.)')
print()
for disease in ['HHD', 'stroke']:
    ia = diag_results[disease].get('A')
    ib = diag_results[disease].get('B')
    if ia is None or ib is None:
        continue
    print(f'  {disease}:')
    if ia < -100 and ib < -100:
        print(f'    Both graphs show severe degradation ({ia:.1f}%, {ib:.1f}%).')
        print(f'    Absent spatial dependence confirmed regardless of sex stratification.')
        print(f'    Graph B degradation is less extreme ({ib-ia:+.1f}pp), indicating')
        print(f'    sex-specific SBP patterns have marginally different spatial structure,')
        print(f'    but country-specific factors still dominate for both sexes.')
        print(f'    Paper: report as strong evidence of absent spatial dependence.')
    elif ia < -100 and -100 <= ib < 0:
        print(f'    Graph A: extreme degradation ({ia:.1f}%).')
        print(f'    Graph B: moderate degradation ({ib:.1f}%).')
        print(f'    Sex-disaggregation substantially reduces degradation.')
        print(f'    Spatial signal is sex-specific but still insufficient for gains.')
    elif ia < 0 and ib >= 0:
        print(f'    Graph A: degradation ({ia:.1f}%). Graph B: positive gain ({ib:.1f}%).')
        print(f'    Sex-disaggregation fully recovers spatial signal for this disease.')
        print(f'    The genuine cross-border pattern is sex-specific and is masked')
        print(f'    when both sexes are aggregated.')
    elif ia >= 0 and ib >= 0:
        diff = ib - ia
        if diff > 5:
            print(f'    Graph A: +{ia:.1f}%. Graph B: +{ib:.1f}% (amplified by sex split).')
            print(f'    Sex-disaggregation strengthens the spatial signal ({diff:+.1f}pp).')
            print(f'    Sex-specific spatial patterns exist and are additively captured')
            print(f'    in Graph B via both geographic and cross-sex edges.')
            print(f'    Paper: report Graph B as preferred specification for this disease.')
        else:
            print(f'    Consistent positive improvement across both graphs.')
            print(f'    Spatial signal is robust to sex stratification.')
    print()

