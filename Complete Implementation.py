#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 15:17:51 2026

@author: julius
"""

# ============================================================
# Trust-Calibrated Intent-Based Ransomware Containment over QUIC
#  Implementation
# ============================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# -------------------------------
# Module 1: Data Generation
# -------------------------------

def generate_quic_flow_data(n_samples=1500, attack_ratio=0.30, random_state=42):
    rng = np.random.default_rng(random_state)
    n_attack = int(n_samples * attack_ratio)
    n_benign = n_samples - n_attack

    benign = pd.DataFrame({
        "packet_rate": rng.normal(115, 40, n_benign).clip(20, 270),
        "avg_pkt_size": rng.normal(820, 210, n_benign).clip(220, 1450),
        "burstiness": rng.normal(0.42, 0.16, n_benign).clip(0.05, 0.90),
        "conn_attempts": rng.poisson(6, n_benign).clip(1, 35),
        "new_cid_rate": rng.normal(0.11, 0.07, n_benign).clip(0.00, 0.48),
        "handshake_fail_ratio": rng.normal(0.07, 0.05, n_benign).clip(0.00, 0.35),
        "host_file_change_rate": rng.normal(38, 18, n_benign).clip(1, 120),
        "dest_diversity": rng.normal(0.33, 0.15, n_benign).clip(0.02, 0.80),
        "label": 0
    })

    attack = pd.DataFrame({
        "packet_rate": rng.normal(190, 70, n_attack).clip(35, 420),
        "avg_pkt_size": rng.normal(700, 260, n_attack).clip(180, 1550),
        "burstiness": rng.normal(0.66, 0.20, n_attack).clip(0.12, 1.00),
        "conn_attempts": rng.poisson(13, n_attack).clip(1, 60),
        "new_cid_rate": rng.normal(0.27, 0.16, n_attack).clip(0.00, 0.85),
        "handshake_fail_ratio": rng.normal(0.20, 0.13, n_attack).clip(0.00, 0.80),
        "host_file_change_rate": rng.normal(118, 58, n_attack).clip(10, 330),
        "dest_diversity": rng.normal(0.59, 0.22, n_attack).clip(0.05, 1.00),
        "label": 1
    })

    df = pd.concat([benign, attack], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    noise_idx = rng.choice(n_samples, size=int(0.04 * n_samples), replace=False)
    df.loc[noise_idx, "label"] = 1 - df.loc[noise_idx, "label"]

    return df.sample(frac=1, random_state=random_state + 1).reset_index(drop=True)
