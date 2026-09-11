# 📱 M-Pesa Fraud Detection

Identifying fraudulent mobile-money transactions using synthetic-but-realistic East African transaction data — modelling the fraud patterns seen in SIM-swap fraud, micro-transaction layering, and velocity anomalies.

## Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Pipeline Architecture](#pipeline-architecture)
- [Methodology](#methodology)
- [Results](#results)
- [Future Improvements](#future-improvements)

---

## Overview

Mobile money is the dominant payment rail across much of East Africa, and fraud patterns on platforms like M-Pesa differ meaningfully from card-based fraud — think SIM-swap takeovers, micro-transaction layering to stay under alert thresholds, and velocity-based anomalies. This project builds a fraud-detection pipeline on synthetic transaction data engineered to reflect those real-world patterns, with an emphasis on outputs a bank or fintech compliance team could actually use.

## Problem Statement

Fraud teams at mobile money operators need to flag suspicious transactions **fast** and **explainably**. A black-box model isn't enough — analysts need to understand *why* a transaction was flagged before acting on it. This project treats explainability as a first-class requirement, not an afterthought.

## Dataset

- Synthetic transaction data generated to reflect realistic East African mobile-money patterns.
- Engineered fraud signals: SIM-swap indicators, transaction velocity, time-of-day anomalies.

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python, SQL |
| Data Handling | Pandas |
| Modelling | XGBoost |
| Explainability | SHAP |
| Feature Engineering | Custom velocity & anomaly features |

## Pipeline Architecture

```
Synthetic Transaction Data Generation
              │
              ▼
        SQL Storage
              │
              ▼
     Python EDA
              │
              ▼
   Feature Engineering
   (velocity, amount deviation, account age)
              │
              ▼
     XGBoost Model Training
              │
              ▼
   SHAP Explainability
              │
              ▼
   Incident-Style Report
```

## Methodology

Key engineered features:
- **Transaction velocity** — number of transactions in the last 5 minutes / 1 hour.
- **Amount deviation** — current transaction size vs. the account's historical average.
- **Time-of-day anomaly** — flags transactions occurring outside a user's typical activity window.
- **Account age** — newer accounts carry higher baseline fraud risk.

Individual predictions are explained using **SHAP waterfall plots**, so each flagged transaction comes with a plain breakdown of which features pushed the risk score up — formatted to be presentable directly to a compliance or fraud-operations team.

## Results

| Metric | Value |
|---|---|
| AUC-ROC | **0.94** |
| Recall @ 1% FPR | **82%** |

SHAP output was used to identify the top fraud indicators driving the model's decisions, producing a report structure directly usable by a bank or mobile-money operator's fraud team.

## Future Improvements

- Extend to a graph-based approach to model networks of related accounts (common in SIM-swap rings).
- Add a streaming/real-time scoring simulation.
- Localise the incident-report template for direct handoff to a fraud-ops team.

