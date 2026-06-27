#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 09:51:01 2026

@author: julius
"""

"""
# ================================
# Module 1: Data Generation
# ================================
import numpy as np
import pandas as pd

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

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
    # Small controlled label noise to reflect uncertainty in real network telemetry.
    noise_idx = rng.choice(n_samples, size=int(0.04 * n_samples), replace=False)
    df.loc[noise_idx, "label"] = 1 - df.loc[noise_idx, "label"]
    return df.sample(frac=1, random_state=random_state + 1).reset_index(drop=True)
df = generate_quic_flow_data()
print("Dataset shape:", df.shape)
print(df.head())
print(df["label"].value_counts())

"""

"""

# ================================
# Module 2: Risk Observer Model
# ================================
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def train_observer_model(df):
    features = [col for col in df.columns if col != "label"]
    X_train, X_test, y_train, y_test = train_test_split(
        df[features],
        df["label"],
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=df["label"]
    )
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=5,
        min_samples_leaf=5,
        random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0)
    }
    return model, features, X_test.reset_index(drop=True), y_test.reset_index(drop=True), y_pred, metrics
model, features, X_test, y_test, y_pred, observer_metrics = train_observer_model(df)
print("Observer model metrics")
for k, v in observer_metrics.items():
    print(f"{k}: {v:.3f}")

"""

"""
# ================================
# Module 3: Trust Calibration
# ================================
def compute_trust_scores(df, model, features):
    risk_probability = model.predict_proba(df[features])[:, 1]

    observable_pressure = (
        0.18 * (df["packet_rate"] / df["packet_rate"].max()) +
        0.12 * (df["conn_attempts"] / df["conn_attempts"].max()) +
        0.18 * df["burstiness"] +
        0.15 * df["new_cid_rate"] +
        0.15 * df["handshake_fail_ratio"] +
        0.22 * (df["host_file_change_rate"] / df["host_file_change_rate"].max())
    ).clip(0, 1)

    risk_score = (0.62 * risk_probability + 0.38 * observable_pressure).clip(0, 1)
    trust_score = (1 - risk_score).clip(0, 1)

    enriched = df.copy()
    enriched["risk_probability"] = risk_probability
    enriched["risk_score"] = risk_score
    enriched["trust_score"] = trust_score

    return enriched

test_df = X_test.copy()
test_df["label"] = y_test
enriched = compute_trust_scores(test_df, model, features)

print(enriched[["risk_probability", "risk_score", "trust_score", "label"]].head(10).round(3))

"""

"""
# ================================
# Module 4: Intent Translation
# ================================
def translate_intent(row):
    if row["trust_score"] < 0.35 and row["risk_score"] >= 0.65:
        return "QUARANTINE_HOST"
    elif row["trust_score"] < 0.58 and row["risk_score"] >= 0.42:
        return "RATE_LIMIT_FLOW"
    return "ALLOW_WITH_MONITORING"
enriched["intent"] = enriched.apply(translate_intent, axis=1)
print("Intent distribution")
print(enriched["intent"].value_counts())
print("\n Intent decisions")
print(
    enriched[
        ["packet_rate", "burstiness", "conn_attempts", "risk_score", "trust_score", "intent"]
    ].head(8).round(3)
)

"""

"""
# ================================
# Module 5: P4-Style Enforcement Simulator
# ================================

def p4_style_enforcement(intent):
    policy_table = {
        "ALLOW_WITH_MONITORING": {
            "action": "forward",
            "rate_limit_mbps": None,
            "priority": 10
        },
        "RATE_LIMIT_FLOW": {
            "action": "meter",
            "rate_limit_mbps": 2,
            "priority": 50
        },
        "QUARANTINE_HOST": {
            "action": "drop_or_redirect_to_sinkhole",
            "rate_limit_mbps": 0,
            "priority": 100
        }
    }
    return policy_table[intent]
enriched["p4_policy"] = enriched["intent"].apply(p4_style_enforcement)
print(enriched[["intent", "p4_policy"]].head(10))

"""


"""
# ================================
# Module 6: Reinforcement Learning Optimiser
# ================================

ACTIONS = ["allow", "rate_limit", "quarantine"]

def state_from_row(row):
    if row["risk_score"] < 0.35:
        risk_bin = 0
    elif row["risk_score"] < 0.65:
        risk_bin = 1
    else:
        risk_bin = 2

    if row["trust_score"] < 0.35:
        trust_bin = 0
    elif row["trust_score"] < 0.65:
        trust_bin = 1
    else:
        trust_bin = 2

    return risk_bin * 3 + trust_bin

def reward_function(action, true_label):
    if true_label == 1:
        return {
            "allow": -10,
            "rate_limit": 4,
            "quarantine": 8
        }[action]

    return {
        "allow": 5,
        "rate_limit": -2,
        "quarantine": -8
    }[action]

def train_q_learning(enriched_df, episodes=50, alpha=0.15, gamma=0.90, epsilon=0.20, random_state=42):
    rng = np.random.default_rng(random_state)
    q_table = np.zeros((9, len(ACTIONS)))
    records = enriched_df.reset_index(drop=True)

    for _ in range(episodes):
        for idx in rng.permutation(len(records)):
            row = records.iloc[idx]
            state = state_from_row(row)

            if rng.random() < epsilon:
                action_idx = rng.integers(len(ACTIONS))
            else:
                action_idx = int(np.argmax(q_table[state]))

            action = ACTIONS[action_idx]
            reward = reward_function(action, int(row["label"]))

            q_table[state, action_idx] += alpha * (
                reward + gamma * np.max(q_table[state]) - q_table[state, action_idx]
            )

    return q_table

q_table = train_q_learning(enriched, episodes=50)

q_df = pd.DataFrame(q_table, columns=ACTIONS)
print("Learned Q-table")
print(q_df.round(2))

"""


"""
# ================================
# Module 7: Evaluation
# ================================
from sklearn.metrics import recall_score
def action_from_baseline(row):
    if row["risk_score"] >= 0.70:
        return "quarantine"
    if row["risk_score"] >= 0.45:
        return "rate_limit"
    return "allow"

def action_from_rl(row, q_table):
    state = state_from_row(row)
    return ACTIONS[int(np.argmax(q_table[state]))]

def evaluate_actions(enriched_df, q_table=None, policy="baseline"):
    rows = []

    for _, row in enriched_df.iterrows():
        if policy == "baseline":
            action = action_from_baseline(row)
        else:
            action = action_from_rl(row, q_table)

        predicted_containment = 1 if action in ["rate_limit", "quarantine"] else 0
        true_label = int(row["label"])

        rows.append({
            "true_label": true_label,
            "action": action,
            "contained": predicted_containment,
            "reward": reward_function(action, true_label)
        })
    result = pd.DataFrame(rows)
    benign_total = max((result["true_label"] == 0).sum(), 1)
    metrics = {
        "containment_recall": recall_score(result["true_label"], result["contained"], zero_division=0),
        "false_positive_rate": ((result["true_label"] == 0) & (result["contained"] == 1)).sum() / benign_total,
        "mean_reward": result["reward"].mean(),
        "quarantine_count": int((result["action"] == "quarantine").sum()),
        "rate_limit_count": int((result["action"] == "rate_limit").sum()),
        "allow_count": int((result["action"] == "allow").sum())
    }
    return result, metrics
baseline_actions, baseline_metrics = evaluate_actions(enriched, policy="baseline")
rl_actions, rl_metrics = evaluate_actions(enriched, q_table=q_table, policy="rl")
print("Baseline policy metrics")
for k, v in baseline_metrics.items():
    print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
print("\nRL policy metrics")
for k, v in rl_metrics.items():
    print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")

"""