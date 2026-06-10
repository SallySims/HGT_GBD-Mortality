# ============================================================
# 04_graph_builders_model_utilities.py
# HGT Spatial Dependence — GBD East Africa
# Graph Builders, HGT Model, Training & Diagnostic Utilities
#
# Part of: HGT_GBD-Mortality
# Repository: https://github.com/SallySims/HGT_GBD-Mortality
#
# Run order: 04 of 11
# Prerequisites: Run files 01 through 03 first
# ============================================================

# ============================================================
# CELL 4: GRAPH BUILDERS, MODEL, TRAINING UTILITIES
# ============================================================

# ── Graph A builder (country × age, 80 nodes) ──
def build_graph_A(df, le_node, spatial=False):
    """
    Nodes  : country × age (80 total).
    Self   : one edge per data row; attrs = [sdi_s*, risk_dummies*, year_s].
    Neighbor: geographic adjacency connecting same-age nodes across
              neighbouring countries (spatial model only).
    * present only when relevant model includes them.
    """
    data = HeteroData()
    data['node'].x = torch.eye(len(le_node.classes_), dtype=torch.float)

    node_ids = le_node.transform(
        [f"{r['location']}|{r['age']}" for _,r in df.iterrows()])
    ei = torch.tensor([node_ids, node_ids], dtype=torch.long)
    data['node','self','node'].edge_index = ei

    attr_cols = [c for c in
        ['sdi_s','risk_BMI','risk_FPG','risk_SBP','year_s'] if c in df.columns]
    data['node','self','node'].edge_attr = torch.tensor(
        df[attr_cols].values.astype(np.float32))
    data['node','self','node'].y = torch.tensor(
        df['y'].values, dtype=torch.float).unsqueeze(1)

    if spatial:
        ages  = sorted(df['age'].unique())
        edges = []
        for age in ages:
            for src, dsts in NEIGHBORS.items():
                sk = f"{src}|{age}"
                for dst in dsts:
                    dk = f"{dst}|{age}"
                    if sk in le_node.classes_ and dk in le_node.classes_:
                        edges.append([le_node.transform([sk])[0],
                                      le_node.transform([dk])[0]])
        if edges:
            data['node','neighbor','node'].edge_index = (
                torch.tensor(edges,dtype=torch.long).t().contiguous())
    return data


# ── Graph B builder (country × sex × age, 160 nodes) ──
def build_graph_B(df, le_node, spatial=False):
    """
    Nodes  : country × sex × age (160 total).
    Self   : one edge per data row.
    Neighbor: geographic adjacency (same sex, same age, neighbouring countries)
              + cross-sex edges (Male ↔ Female, same country, same age).
    """
    data = HeteroData()
    data['node'].x = torch.eye(len(le_node.classes_), dtype=torch.float)

    node_ids = le_node.transform(
        [f"{r['location']}|{r['sex']}|{r['age']}" for _,r in df.iterrows()])
    ei = torch.tensor([node_ids, node_ids], dtype=torch.long)
    data['node','self','node'].edge_index = ei

    attr_cols = [c for c in
        ['sdi_s','risk_BMI','risk_FPG','risk_SBP','year_s'] if c in df.columns]
    data['node','self','node'].edge_attr = torch.tensor(
        df[attr_cols].values.astype(np.float32))
    data['node','self','node'].y = torch.tensor(
        df['y'].values, dtype=torch.float).unsqueeze(1)

    if spatial:
        ages  = sorted(df['age'].unique())
        sexes = sorted(df['sex'].unique())
        edges = []
        # Geographic adjacency
        for sex in sexes:
            for age in ages:
                for src, dsts in NEIGHBORS.items():
                    sk = f"{src}|{sex}|{age}"
                    for dst in dsts:
                        dk = f"{dst}|{sex}|{age}"
                        if sk in le_node.classes_ and dk in le_node.classes_:
                            edges.append([le_node.transform([sk])[0],
                                          le_node.transform([dk])[0]])
        # Cross-sex edges
        for country in COUNTRIES:
            for age in ages:
                mk = f"{country}|Male|{age}"
                fk = f"{country}|Female|{age}"
                if mk in le_node.classes_ and fk in le_node.classes_:
                    edges.append([le_node.transform([mk])[0],
                                  le_node.transform([fk])[0]])
                    edges.append([le_node.transform([fk])[0],
                                  le_node.transform([mk])[0]])
        if edges:
            data['node','neighbor','node'].edge_index = (
                torch.tensor(edges,dtype=torch.long).t().contiguous())
    return data


# ── HGT Model ──
class SpilloverHGT(nn.Module):
    """
    Architecture:
      1. Type-specific linear projection → HIDDEN dim
      2. Two stacked HGTConv layers, HEADS attention heads each
      3. ReLU after each conv layer
      4. Edge-level MLP: [node_embed || edge_attr] → scalar prediction
         Dropout(DROPOUT) applied for regularisation (weight_decay in optimiser)
    """
    def __init__(self, hidden=HIDDEN, heads=HEADS, dropout=DROPOUT):
        super().__init__()
        self.hidden=hidden; self.dropout=dropout
        self.proj=None; self.convs=nn.ModuleList(); self.mlp=None
        self._attention_weights = {}   # store for visualisation

    def _init_lazy(self, data, device):
        if self.proj is not None: return
        self.proj = nn.ModuleDict({
            n: Linear(data[n].x.size(1), self.hidden).to(device)
            for n in data.node_types})
        for _ in range(2):
            self.convs.append(
                HGTConv(self.hidden, self.hidden,
                        data.metadata(), heads=HEADS).to(device))
        edge_dim = data['node','self','node'].edge_attr.size(1)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden+edge_dim, self.hidden),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden,1)).to(device)

    def forward(self, data):
        device = data['node','self','node'].y.device
        self._init_lazy(data, device)
        x = {k: self.proj[k](data[k].x) for k in self.proj}
        for i, conv in enumerate(self.convs):
            x = conv(x, data.edge_index_dict)
            x = {k: F.relu(v) for k,v in x.items()}
        ei = data['node','self','node'].edge_index
        ea = data['node','self','node'].edge_attr
        src,_ = ei
        return self.mlp(torch.cat([x['node'][src], ea], dim=1))


# ── Single-seed train ──
def train_model(train_data, test_data, scaler, seed,
                epochs=EPOCHS, track_grads=False):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = SpilloverHGT().to(device)
    train_data = train_data.to(device)
    test_data  = test_data.to(device)
    _ = model(train_data)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    loss_curve = []
    grad_norms  = []
    final_loss  = None
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(train_data)
        loss = loss_fn(pred, train_data['node','self','node'].y)
        loss.backward()
        if track_grads:
            gn = sum(p.grad.norm().item()**2
                     for p in model.parameters() if p.grad is not None)**0.5
            grad_norms.append(gn)
        optimizer.step()
        final_loss = loss.item()
        loss_curve.append(final_loss)
    model.eval()
    with torch.no_grad():
        ps = model(test_data).cpu().numpy().flatten()
        ts = test_data['node','self','node'].y.cpu().numpy().flatten()
        po = scaler.inverse_transform(ps.reshape(-1,1)).flatten()
        to = scaler.inverse_transform(ts.reshape(-1,1)).flatten()
        residuals = to - po
        mse = mean_squared_error(to, po)
        r2  = r2_score(to, po)
    return mse, r2, final_loss, residuals, loss_curve, grad_norms, model


# ── Multi-seed wrapper ──
def run_multiseed(train_fn, test_fn, scaler, seeds=SEEDS, track_grads=False):
    mses,r2s,losses,curves,gnorms = [],[],[],[],[]
    last_resid = None; last_model = None
    for seed in seeds:
        mse,r2,loss,resid,curve,gn,mdl = train_model(
            train_fn(), test_fn(), scaler, seed, track_grads=track_grads)
        mses.append(mse); r2s.append(r2); losses.append(loss)
        curves.append(curve); gnorms.append(gn)
        last_resid=resid; last_model=mdl
    return dict(
        mse_mean=np.mean(mses), mse_sd=np.std(mses),
        r2_mean =np.mean(r2s),  r2_sd =np.std(r2s),
        loss_mean=np.mean(losses),loss_sd=np.std(losses),
        residuals=last_resid, loss_curves=curves,
        grad_norms=gnorms,    model=last_model)


# ── Moran's I ──
def morans_i(residuals, test_df):
    tmp = test_df.iloc[:len(residuals)].copy()
    tmp['resid'] = residuals
    cr  = tmp.groupby('location')['resid'].mean()
    idx = {c:i for i,c in enumerate(COUNTRIES)}
    n   = len(COUNTRIES)
    W   = np.zeros((n,n))
    for s,ds in NEIGHBORS.items():
        for d in ds:
            if s in idx and d in idx: W[idx[s],idx[d]]=1.0
    rs=W.sum(axis=1,keepdims=True); rs[rs==0]=1; W=W/rs
    y  = np.array([cr.get(c,0.0) for c in COUNTRIES])
    S0 = W.sum(); yc=y-y.mean()
    I  = (n/S0)*(yc@W@yc)/(yc@yc) if (yc@yc)!=0 else 0.0
    sim=[]
    for _ in range(999):
        yp=np.random.permutation(yc)
        sim.append((n/S0)*(yp@W@yp)/(yp@yp) if (yp@yp)!=0 else 0.0)
    p=np.mean(np.abs(sim)>=np.abs(I))
    return round(I,4), round(p,4)


# ── Bootstrap CI ──
def boot_ci(bm,bs,sm,ss,n=len(SEEDS),n_boot=1000):
    diffs=[]
    for _ in range(n_boot):
        b=np.mean(np.random.normal(bm,bs+1e-9,n))
        s=np.mean(np.random.normal(sm,ss+1e-9,n))
        diffs.append((b-s)/b*100 if b!=0 else 0.0)
    return round(np.percentile(diffs,2.5),2), round(np.percentile(diffs,97.5),2)


# ── Spatial lag OLS benchmark ──
def spatial_lag_ols(train_df, test_df, feat_cols):
    idx={c:i for i,c in enumerate(COUNTRIES)}; n=len(COUNTRIES)
    W=np.zeros((n,n))
    for s,ds in NEIGHBORS.items():
        for d in ds:
            if s in idx and d in idx: W[idx[s],idx[d]]=1.0
    rs=W.sum(axis=1,keepdims=True); rs[rs==0]=1; W=W/rs
    def add_lag(df):
        cm=df.groupby('location')['deathratevalue'].mean()
        yv=np.array([cm.get(c,0.0) for c in COUNTRIES])
        wy=W@yv
        df=df.copy()
        df['spatial_lag']=df['location'].map(
            {c:wy[i] for i,c in enumerate(COUNTRIES)})
        return df
    tr=add_lag(train_df); te=add_lag(test_df)
    af=feat_cols+['spatial_lag']
    ols=LinearRegression().fit(tr[af].fillna(0).values,
                               tr['deathratevalue'].values)
    pred=ols.predict(te[af].fillna(0).values)
    return (round(mean_squared_error(te['deathratevalue'].values,pred),4),
            round(r2_score(te['deathratevalue'].values,pred),3))


# ── CAR benchmark (Conditional Autoregressive) ──
def car_benchmark(train_df, test_df, feat_cols):
    """
    CAR model via spatial 2SLS (spreg.GM_Lag).
    Traditional spatial econometric benchmark alongside OLS spatial lag.
    """
    try:
        from spreg import GM_Lag
        idx={c:i for i,c in enumerate(COUNTRIES)}; n=len(COUNTRIES)
        W_arr=np.zeros((n,n))
        for s,ds in NEIGHBORS.items():
            for d in ds:
                if s in idx and d in idx: W_arr[idx[s],idx[d]]=1.0
        rs=W_arr.sum(axis=1,keepdims=True); rs[rs==0]=1; W_arr=W_arr/rs
        W_sp = libpysal.weights.full2W(W_arr)

        # Aggregate to country level for CAR
        tr_c = train_df.groupby('location').agg(
            deathratevalue=('deathratevalue','mean'),
            **{f:('mean') for f in feat_cols if f in train_df.columns}
        ).reindex(COUNTRIES).fillna(0)
        te_c = test_df.groupby('location').agg(
            deathratevalue=('deathratevalue','mean'),
            **{f:('mean') for f in feat_cols if f in test_df.columns}
        ).reindex(COUNTRIES).fillna(0)

        y_tr = tr_c['deathratevalue'].values.reshape(-1,1)
        X_tr = tr_c[[f for f in feat_cols if f in tr_c.columns]].values
        if X_tr.shape[1]==0:
            return None, None
        model = GM_Lag(y_tr, X_tr, w=W_sp, name_y='deathratevalue')
        # Predict on test (OLS part only — GM_Lag doesn't have predict())
        b = model.betas.flatten()
        X_te = te_c[[f for f in feat_cols if f in te_c.columns]].values
        X_te_aug = np.column_stack([np.ones(len(X_te)), X_te])
        if X_te_aug.shape[1] == len(b):
            pred = X_te_aug @ b
            y_te = te_c['deathratevalue'].values
            return (round(mean_squared_error(y_te,pred),4),
                    round(r2_score(y_te,pred),3))
        return None, None
    except Exception as e:
        print(f"    CAR model skipped: {e}")
        return None, None


print("All graph/model/training utilities defined.")
