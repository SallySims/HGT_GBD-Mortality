# ============================================================
# 01_setup.py
# HGT Spatial Dependence — GBD East Africa
# Setup & Dependencies
#
# Part of: HGT_GBD-Mortality
# Repository: https://github.com/SallySims/HGT_GBD-Mortality
#
# Run order: 01 of 11
# Prerequisites: Run files 01 through 00 first
# ============================================================

# ============================================================
# CELL 1: SETUP & DEPENDENCIES
# ============================================================

!pip install torch_geometric pysal libpysal esda spreg -q

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, warnings, copy
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

import libpysal
from esda.moran import Moran

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
else:
    print("Device  : CPU")

# ── Hyperparameters ──
SEEDS             = [42, 123, 456, 789, 1024]
EPOCHS            = 200
HIDDEN            = 64
HEADS             = 4
LR                = 0.003
WEIGHT_DECAY      = 1e-4
DROPOUT           = 0.2
TRAIN_YEAR_CUTOFF = 2015
COUNTRIES         = ["Burundi","Kenya","Rwanda","Tanzania","Uganda"]
RISK_LIST         = ["BMI","FPG","SBP"]
MC_SAMPLES        = 50   # Monte Carlo uncertainty propagation runs

ROLLING_WINDOWS = [
    (1990,2000,2001,2005,"W1"),
    (1990,2005,2006,2010,"W2"),
    (1990,2010,2011,2015,"W3"),
    (1990,2015,2016,2023,"W4-original"),
]

NEIGHBORS = {
    "Uganda"  :["Kenya","Rwanda","Tanzania"],
    "Rwanda"  :["Uganda","Burundi","Tanzania"],
    "Kenya"   :["Uganda","Tanzania"],
    "Tanzania":["Uganda","Kenya","Rwanda","Burundi"],
    "Burundi" :["Rwanda","Tanzania"]
}

print("\nHyperparameters:")
for k,v in dict(HIDDEN=HIDDEN,HEADS=HEADS,LR=LR,
                WEIGHT_DECAY=WEIGHT_DECAY,DROPOUT=DROPOUT,
                EPOCHS=EPOCHS,SEEDS=SEEDS,MC_SAMPLES=MC_SAMPLES).items():
    print(f"  {k:18s}: {v}")
