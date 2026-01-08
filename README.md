# XGBoost for Survival Analysis in Customer Churn Prediction

## Project Overview

This repository presents an advanced methodology for **Customer Churn Prediction**, integrating the capabilities of **eXtreme Gradient Boosting (XGBoost)** with **Survival Analysis**. Unlike standard classification models that only predict *if* a customer leaves, this framework models *when* the event will occur, accounting for temporal dependencies and **data censoring**.

Standard models often ignore customers who are still active at the time of analysis (right-censoring), leading to biased results. This project bridges the gap between classical statistics and modern machine learning.

## Why C++ for Survival Analysis?

* **Maximum Performance:** C++ ensures the highest execution speed for training on massive datasets.


* **Quant Integration:** It allows seamless integration into high-frequency financial or "Quant" pipelines.


* **Memory Efficiency:** It provides fine-grained control over resources when handling millions of patient or customer records.


* **Scalability:** The architecture is designed to scale across parallel computing resources effectively.



## Key Features

* **Advanced Objective Functions:** Implementation of **Cox Proportional Hazards** (Partial Likelihood) and **Accelerated Failure Time (AFT)** models.


* **Mathematical Precision:** Utilizes **2nd-order Taylor Expansion** to approximate the loss function, providing more stable and accurate updates than standard gradient boosting.


* **Sparsity Awareness:** Built-in handling of missing values, eliminating the need for complex pre-processing.


* **Regularization:** Integrated **L1 (Lasso)** and **L2 (Ridge)** penalties to prevent overfitting in high-dimensional datasets.



## Methodology Highlights

### 1. Handling Censored Data

We define the event time  as the minimum of the true event time  and the censoring time . The model optimizes the log-likelihood to include both events and censored observations.

### 2. The Power of the Hessian

XGBoost leverages the second derivative (Hessian) of the loss function. This allows the algorithm to understand the "curvature" of the error landscape, learning aggressively in flat areas and cautiously in steep ones.

## Repository Structure

```text
├── cpp/                # High-performance C++ implementation for Quant pipelines
├── python/             # Industry-standard implementation using the xgboost library
├── r_scripts/          # Statistical analysis and Kaplan-Meier comparisons
├── paper/              # "XGBoost in Survival Analysis" (PDF)
└── data/               # Synthetic Telco Churn dataset examples

```

## Performance Comparison

The included AFT (Accelerated Failure Time) models demonstrate superior forecasting capabilities for events outside the observation window compared to traditional linear models.
