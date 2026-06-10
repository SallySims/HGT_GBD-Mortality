# ============================================================
# 09_monte_carlo_uncertainty.py
# HGT Spatial Dependence — GBD East Africa
# GBD Monte Carlo Uncertainty Propagation
#
# Part of: HGT_GBD-Mortality
# Repository: https://github.com/SallySims/HGT_GBD-Mortality
#
# Run order: 09 of 11
# Prerequisites: Run files 01 through 08 first
# ============================================================

# ============================================================
# CELL 9: GBD MONTE CARLO UNCERTAINTY PROPAGATION
# Addresses GBD measurement uncertainty concern (Reviewer 2)
# ============================================================
# GBD provides upper/lower uncertainty bounds for all estimates.
# These bounds are ASYMMETRIC (upper_diff ~ 1.5x lower_diff on average),
# so we sample from a log-normal distribution parameterised from
# the GBD intervals rather than a symmetric normal.
#
# SDI lower_value == upper_value == mean_value in this dataset,
# so uncertainty propagation applies to deathratevalue only.
#
# Procedure:
#   1. For each MC run, sample deathratevalue from log-normal(mu, sigma)
#      where mu/sigma are derived from each row's point estimate + bounds.
#   2. Train and evaluate the full spatial HGT model on the sampled data.
#   3. After MC_SAMPLES runs, decompose variance into:
#      - GBD uncertainty variance (across MC samples, fixed seed)
#      - Model randomness variance (across seeds, fixed point estimate)
# ============================================================

def lognormal_params_from_bounds(mu_val, lower, upper):
    """
    Derive log-normal mu and sigma from GBD point estimate and bounds.
    Assumes GBD bounds approximate a 95% uncertainty interval.
    Clips to avoid log(0).
    """
    mu_val = np.maximum(mu_val, 1e-6)
    lower  = np.maximum(lower,  1e-6)
    upper  = np.maximum(upper,  mu_val)  # ensure upper >= mu
    log_mu    = np.log(mu_val)
    log_sigma = (np.log(upper) - np.log(lower)) / (2 * 1.96)
    log_sigma = np.maximum(log_sigma, 1e-6)
    return log_mu, log_sigma

def sample_gbd_data(df):
    """
    Return a copy of df with deathratevalue replaced by one
    log-normal sample per row, drawn from its GBD uncertainty interval.
    """
    df_s = df.copy()
    log_mu, log_sigma = lognormal_params_from_bounds(
        df['deathratevalue'].values,
        df['lower'].values,
        df['upper'].values)
    df_s['deathratevalue'] = np.random.lognormal(log_mu, log_sigma)
    return df_s


def run_mc_uncertainty(df_in, disease, build_fn, node_key_fn,
                       use_sdi, use_risk, tag,
                       n_mc=MC_SAMPLES, seed_fixed=42):
    """
    Run MC uncertainty propagation for one disease.
    Returns:
      - mc_improvements: list of improvement % per MC sample
      - model_improvements: list of improvement % per seed (point estimate)
      - variance decomposition
    """
    df_d = df_in[df_in['cause']==disease].copy()

    def _run_one(df_use, seed):
        train_df = df_use[df_use['year']<=TRAIN_YEAR_CUTOFF].copy()
        test_df  = df_use[df_use['year'] >TRAIN_YEAR_CUTOFF].copy()
        if len(train_df)==0 or len(test_df)==0: return None

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
        if use_risk and 'risk_factor' in df_use.columns:
            for r in RISK_LIST:
                train_df[f'risk_{r}']=(train_df['risk_factor']==r).astype(float)
                test_df[f'risk_{r}'] =(test_df['risk_factor'] ==r).astype(float)

        le=LabelEncoder()
        ak=sorted(set(train_df.apply(node_key_fn,axis=1))
                | set(test_df.apply(node_key_fn,axis=1)))
        le.fit(ak)

        mse_b,_,_,_,_,_,_ = train_model(
            build_fn(train_df,le,spatial=False),
            build_fn(test_df, le,spatial=False), t_sc, seed=seed)
        mse_s,_,_,_,_,_,_ = train_model(
            build_fn(train_df,le,spatial=True),
            build_fn(test_df, le,spatial=True),  t_sc, seed=seed)

        return (mse_b-mse_s)/mse_b*100 if mse_b>0 else 0.0

    # MC: vary data, fix seed
    np.random.seed(0)
    mc_improvements = []
    for i in range(n_mc):
        df_sampled = sample_gbd_data(df_d)
        impr = _run_one(df_sampled, seed=seed_fixed)
        if impr is not None:
            mc_improvements.append(impr)
        if (i+1) % 10 == 0:
            print(f"    MC {i+1}/{n_mc} done", end='\r')
    print()

    # Model variance: vary seed, fix data (point estimate)
    model_improvements = []
    for seed in SEEDS:
        impr = _run_one(df_d, seed=seed)
        if impr is not None:
            model_improvements.append(impr)

    mc_arr  = np.array(mc_improvements)
    mdl_arr = np.array(model_improvements)

    var_gbd   = np.var(mc_arr)
    var_model = np.var(mdl_arr)
    total_var = var_gbd + var_model
    pct_gbd   = var_gbd/total_var*100   if total_var>0 else 0
    pct_model = var_model/total_var*100 if total_var>0 else 0

    print(f"  {disease} [{tag}]:")
    print(f"    Point estimate improvement : {np.mean(model_improvements):.2f}%")
    print(f"    MC mean improvement        : {mc_arr.mean():.2f}% "
          f"± {mc_arr.std():.2f}%  "
          f"95%CI [{np.percentile(mc_arr,2.5):.2f}, {np.percentile(mc_arr,97.5):.2f}]")
    print(f"    Variance from GBD uncertainty : {var_gbd:.4f} ({pct_gbd:.1f}%)")
    print(f"    Variance from model randomness: {var_model:.4f} ({pct_model:.1f}%)")

    return mc_arr, mdl_arr, {
        'Disease':disease,'Graph':tag,
        'Point_Impr':round(np.mean(model_improvements),2),
        'MC_Mean':round(mc_arr.mean(),2),
        'MC_SD':round(mc_arr.std(),2),
        'MC_CI_Lo':round(np.percentile(mc_arr,2.5),2),
        'MC_CI_Hi':round(np.percentile(mc_arr,97.5),2),
        'Var_GBD':round(var_gbd,4),
        'Var_Model':round(var_model,4),
        'Pct_GBD':round(pct_gbd,1),
        'Pct_Model':round(pct_model,1)
    }


print("Running GBD Monte Carlo uncertainty propagation (Full model, Graph A)...")
print(f"MC samples per disease: {MC_SAMPLES}")

dfA_mc = dfA[dfA['risk_factor'].isin(RISK_LIST)][
    ['location','age','cause','year','deathratevalue',
     'SDI_Quintile','risk_factor','age_mid','upper','lower']].dropna().copy()

mc_summary_rows = []
mc_all = {}

for disease in sorted(dfA_mc['cause'].unique()):
    mc_arr, mdl_arr, row = run_mc_uncertainty(
        dfA_mc, disease, build_graph_A,
        lambda r: f"{r['location']}|{r['age']}",
        use_sdi=True, use_risk=True, tag="GraphA-Full")
    mc_summary_rows.append(row)
    mc_all[disease] = (mc_arr, mdl_arr)

mc_df = pd.DataFrame(mc_summary_rows)
print("\nMonte Carlo Uncertainty Summary:")
print(mc_df.to_string(index=False))

# Plot distributions
fig, axes = plt.subplots(1, len(mc_all), figsize=(4*len(mc_all),4))
if len(mc_all)==1: axes=[axes]
for ax, (dis,(mc_arr,mdl_arr)) in zip(axes, mc_all.items()):
    ax.hist(mc_arr, bins=20, alpha=0.6, color='steelblue', label='GBD MC')
    ax.hist(mdl_arr,bins=10, alpha=0.6, color='tomato',   label='Model seeds')
    ax.axvline(0, color='black', linestyle='--', linewidth=0.8)
    ax.set_title(dis); ax.set_xlabel('Improvement %'); ax.legend(fontsize=7)
plt.suptitle("Improvement % Distribution: GBD Uncertainty vs Model Randomness",
             fontsize=11)
plt.tight_layout()
plt.savefig('mc_uncertainty_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: mc_uncertainty_distributions.png")
