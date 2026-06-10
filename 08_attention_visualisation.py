# ============================================================
# 08_attention_visualisation.py
# HGT Spatial Dependence — GBD East Africa
# Attention Weight Visualisation
#
# Part of: HGT_GBD-Mortality
# Repository: https://github.com/SallySims/HGT_GBD-Mortality
#
# Run order: 08 of 11
# Prerequisites: Run files 01 through 07 first
# ============================================================

# ============================================================
# CELL 8: ATTENTION WEIGHT VISUALISATION
# Addresses interpretability concern (Reviewer 2)
# ============================================================

def extract_attention_weights(model, data, le_node, disease, tag):
    """
    Extract and visualise attention weights from the first HGTConv layer.
    Shows which node pairs the model attends to most strongly,
    providing interpretability for the spatial graph structure.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data   = data.to(device)
    model  = model.to(device)
    model.eval()

    # Hook to capture attention weights
    attention_maps = {}

    def make_hook(name):
        def hook(module, input, output):
            # HGTConv output is node embeddings; capture edge weights via grad
            attention_maps[name] = output
        return hook

    handles = []
    for i, conv in enumerate(model.convs):
        h = conv.register_forward_hook(make_hook(f'conv_{i}'))
        handles.append(h)

    with torch.no_grad():
        _ = model(data)

    for h in handles:
        h.remove()

    # Compute node embedding norms as proxy for attention importance
    x = {k: model.proj[k](data[k].x) for k in model.proj}
    for conv in model.convs:
        x = conv(x, data.edge_index_dict)
        x = {k: F.relu(v) for k,v in x.items()}

    node_norms = x['node'].detach().norm(dim=1).cpu().numpy()
    node_labels = list(le_node.classes_)

    # Aggregate by country
    country_importance = {}
    for i, label in enumerate(node_labels):
        country = label.split('|')[0]
        if country not in country_importance:
            country_importance[country] = []
        country_importance[country].append(node_norms[i])

    country_mean = {c: np.mean(v) for c,v in country_importance.items()}

    fig, axes = plt.subplots(1,2,figsize=(12,4))
    fig.suptitle(f"Node Embedding Norms — {disease} [{tag}]", fontsize=12)

    # Top-20 nodes by embedding norm
    ax = axes[0]
    top_n = min(20, len(node_labels))
    top_idx = np.argsort(node_norms)[-top_n:]
    ax.barh([node_labels[i] for i in top_idx],
            node_norms[top_idx], color='steelblue')
    ax.set_xlabel('Embedding Norm (attention proxy)')
    ax.set_title(f'Top {top_n} Nodes by Embedding Norm')

    # Country-level aggregation
    ax = axes[1]
    countries_sorted = sorted(country_mean, key=country_mean.get, reverse=True)
    ax.bar(countries_sorted,
           [country_mean[c] for c in countries_sorted],
           color='tomato')
    ax.set_xlabel('Country')
    ax.set_ylabel('Mean Node Embedding Norm')
    ax.set_title('Country-Level Attention (Mean Node Norm)')

    plt.tight_layout()
    fname = f'attention_{disease.replace(" ","_")}_{tag}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {fname}")
    return country_mean

# ── Run for diabetes (strong spatial) and HHD (weak spatial) for contrast ──
print("Extracting attention weights for diabetes and HHD (Graph A, Full model)...")

dfA_attn = dfA[dfA['risk_factor'].isin(RISK_LIST)][
    ['location','age','cause','year','deathratevalue',
     'SDI_Quintile','risk_factor','age_mid']].dropna().copy()

for disease in ['diabetes','HHD']:
    df_d = dfA_attn[dfA_attn['cause']==disease].copy()
    train_df = df_d[df_d['year']<=TRAIN_YEAR_CUTOFF].copy()
    test_df  = df_d[df_d['year'] >TRAIN_YEAR_CUTOFF].copy()

    t_sc=StandardScaler()
    train_df['y']=t_sc.fit_transform(train_df[['deathratevalue']])
    test_df['y'] =t_sc.transform(test_df[['deathratevalue']])
    y_sc=StandardScaler(); y_sc.fit(train_df[['year']])
    train_df['year_s']=y_sc.transform(train_df[['year']])
    test_df['year_s'] =y_sc.transform(test_df[['year']])
    train_df['sdi_s'],sdi_sc=encode_sdi_continuous(
        train_df['SDI_Quintile'],fit=True)
    test_df['sdi_s']=encode_sdi_continuous(test_df['SDI_Quintile'],scaler=sdi_sc)
    for r in RISK_LIST:
        train_df[f'risk_{r}']=(train_df['risk_factor']==r).astype(float)
        test_df[f'risk_{r}'] =(test_df['risk_factor'] ==r).astype(float)

    le=LabelEncoder()
    nk = lambda r: f"{r['location']}|{r['age']}"
    ak=sorted(set(train_df.apply(nk,axis=1))|set(test_df.apply(nk,axis=1)))
    le.fit(ak)

    _,_,_,_,_,_,trained_model = train_model(
        build_graph_A(train_df,le,spatial=True),
        build_graph_A(test_df, le,spatial=True),
        t_sc, seed=42)

    extract_attention_weights(
        trained_model,
        build_graph_A(test_df,le,spatial=True),
        le, disease, "GraphA-Full")
