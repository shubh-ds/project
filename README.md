# A Full-Stack Real Estate Analytics Platform for the Mumbai Market

## Overview

This repository contains the complete codebase for a full-stack decision support system designed to navigate the complexities of the Mumbai and Thane real estate markets. The project moves beyond standard "black box" price prediction to deliver a holistic, transparent and user-centric tool.

The core mission is to empower prospective homebuyers by answering three critical questions:
1.  **What is a fair, data-driven price for this property?** (Prediction)
2.  **How confident can I be in that price?** (Uncertainty Quantification)
3.  **Why was this price predicted, and what are my best alternatives?** (Explainability & Recommendation)

This project covers the entire data science lifecycle from data enrichment and feature engineering to model deployment in an interactive web application.

## Key Features

*   **Advanced Feature Engineering:**
    *   Integration of **spatial features** by calculating distances to 13 key points of interest (POIs) across Mumbai and Thane.
    *   Development of a novel **"Society DNA"** concept, creating a rich feature vector for each residential society to power recommendations.

*   **High-Accuracy Predictive Modeling:**
    *   Systematic benchmarking of **18 machine learning algorithms** across 4 different preprocessing strategies.
    *   Implementation of a custom **Stacked Ensemble Regressor** that combines the top 5 models to achieve state-of-the-art accuracy.

*   **Uncertainty Quantification:**
    *   Implementation of **Conformal Prediction** to generate statistically robust 95% confidence intervals for every price estimate, providing a reliable measure of the model's confidence.

*   **Model Explainability (XAI):**
    *   Integration of **SHAP (SHapley Additive exPlanations)** to provide full model transparency, allowing for the interpretation of both global feature importances and the drivers of individual predictions.

*   **Intelligent Recommendation System:**
    *   A content-based filtering system that uses the "Society DNA" and **Cosine Similarity** to recommend societies that are most similar to a user's selection.

*   **Interactive Web Application:**
    *   A fully functional front-end built with **Dash by Plotly** that integrates all backend components into an intuitive user interface.

## Tech Stack

*   **Data Analysis & Manipulation:** Pandas, NumPy
*   **Machine Learning & Preprocessing:** Scikit-learn, CatBoost, XGBoost, LightGBM
*   **Model Explainability:** SHAP
*   **Web Application:** Dash, Plotly
*   **Data Serialization:** Joblib
*   **Development Environment:** Jupyter Notebooks, Python 3.12