# ============================================================
# 02_data_loading.py
# HGT Spatial Dependence — GBD East Africa
# Data Loading & Preprocessing
#
# Part of: HGT_GBD-Mortality
# Repository: https://github.com/SallySims/HGT_GBD-Mortality
#
# Run order: 02 of 11
# Prerequisites: Run files 01 through 01 first
# ============================================================

# ============================================================
# CELL 2: DATA LOADING & PREPROCESSING
# ============================================================

from google.colab import drive
drive.mount('/content/drive')
os.chdir('/content/drive/MyDrive/MyProjects/GBDProject/Transformer(Graph)')

df1 = pd.read_csv('GBD_DataAll.csv')
df2 = pd.read_csv('SDI_GBD.csv')
df1['age'] = df1['age'].str.strip()

# ── Fix: negative deathratevalue in stroke young age groups ──
# GBD modelling artefact (2,211 rows, all stroke, ages 20-54).
# Clip to zero — these represent near-zero rates with wide uncertainty bands.
neg_count = (df1['deathratevalue'] < 0).sum()
df1['deathratevalue'] = df1['deathratevalue'].clip(lower=0)
df1['lower']          = df1['lower'].clip(lower=0)
print(f"Clipped {neg_count} negative deathratevalue rows to 0 (stroke, young ages)")

# ── Fix: merge SDI on location+year only ──
# SDI file has sex=Both, age=All Ages only.
# Original merge on sex left Male/Female rows with null SDI.
# Broadcasting on location+year fixes this.
df2_sdi = (df2[['location','year','SDI_Quintile','mean_value']]
           .rename(columns={'mean_value':'sdi_value'})
           .drop_duplicates(subset=['location','year']))

# Note: SDI lower_value == upper_value == mean_value in this dataset,
# so no SDI-level uncertainty propagation is possible.

# ── Age midpoint ──
def age_mid(x):
    """'85+' → 90 (representative for open-ended group in low-LE settings)."""
    x = str(x).strip()
    if '+' in x:  return 90.0
    if '-' in x:  return np.mean([float(i) for i in x.split('-')])
    return float(x)

# ── Graph A: Both sex ──
dfA = df1[df1['sex']=='Both'].copy()
dfA = pd.merge(dfA, df2_sdi, on=['location','year'], how='left')
dfA['age_mid'] = dfA['age'].apply(age_mid)

# ── Graph B: Male + Female ──
dfB = df1[df1['sex'].isin(['Male','Female'])].copy()
dfB = pd.merge(dfB, df2_sdi, on=['location','year'], how='left')
dfB['age_mid'] = dfB['age'].apply(age_mid)

print(f"Graph A: {dfA.shape}  nodes={dfA[['location','age']].drop_duplicates().shape[0]}")
print(f"Graph B: {dfB.shape}  nodes={dfB[['location','sex','age']].drop_duplicates().shape[0]}")
print(f"SDI nulls A: {dfA['SDI_Quintile'].isna().sum()}  B: {dfB['SDI_Quintile'].isna().sum()}")

# ── SDI continuous encoding ──
SDI_ORDER = {q:i for i,q in enumerate(sorted(df2['SDI_Quintile'].dropna().unique()))}
print(f"SDI order: {SDI_ORDER}")

def encode_sdi_continuous(series, scaler=None, fit=False):
    """Ordinal rank → StandardScaler. Preserves SDI's continuous scale."""
    ranked = series.map(SDI_ORDER).values.reshape(-1,1).astype(np.float32)
    if fit:
        sc = StandardScaler()
        return sc.fit_transform(ranked).flatten(), sc
    return scaler.transform(ranked).flatten()
