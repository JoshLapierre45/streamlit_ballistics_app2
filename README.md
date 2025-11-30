Ballistics Explorer – Streamlit Web Application
Hit Probability Modeling • EDA Gallery • Dashboard • Interactive Simulator
1. Overview

Ballistics Explorer is an interactive Streamlit web application designed to explore long-range shooting behavior through data, simulation, and machine learning. The app is organized into four pages, each serving a different analytical purpose:

ML Hit Probability (Lightweight Model) – A fast, pre-trained logistic regression model predicting probability of hit using simplified ballistics inputs.

Hit Probability (Full Calibrated Model) – A user-driven modeling interface supporting dataset uploads, full feature engineering, model calibration, and SHAP explanations.

EDA Gallery – Exploratory data visualization (distributions, relationships, ballistics patterns, etc.) based on the chosen dataset.

Dashboard Page – A polished, dashboard-style interface summarizing key ballistics metrics and trends with insights.

The goal of this project is to give users an intuitive, data-driven understanding of how range, muzzle velocity, wind, shooter dispersion, and environmental factors influence hit probability.

2. Page Summaries
Page 1 — ML Hit Probability (Lightweight Model)

A simplified user interface that uses a pretrained logistic regression model for fast predictions.
Users provide:

Range (yd)

Muzzle velocity & SD

Group size (MOA)

Target size (MOA)

Wind speed

DA, temperature, pressure, humidity

Outputs:

Estimated hit probability (numeric + progress bar)

Probability vs Range line chart

A simulated 5-shot group visualization

Real-time interactivity for exploring ballistics behavior

This page is designed for speed, clarity, and lightweight inference.

Page 2 — Hit Probability (Full Calibrated Model)

This page implements a full machine-learning workflow:

-User can upload shooting history

Accepts per-shot data or summary aggregates

Performs schema validation

Expands aggregates to individual shots

Cleans, coerces, and validates data

-Full feature engineering system

Crosswind & headwind components

Interaction terms (range × wind, range × target, group × range)

Squared terms for nonlinear effects

Rest system encoding (bench, prone, tripod, etc.)

-Model Training

Trains a logistic regression + StandardScaler + CalibratedClassifierCV pipeline:

Isotonic or sigmoid calibration

Train/test split with stratification

Balanced class weights

Robust fallback logic for small datasets

-Model Evaluation

Log Loss is computed and displayed

Ensures evaluation integrity

Warns user if dataset too small for holdout

-Explanation Layer

SHAP feature-importance bar plot

Shows which features most strongly influence predicted hit probability

-Predict Hit % vs Range

Allows controlled predictions across a range band

Generates p(hit) vs distance chart

Displays prediction table with percentages

This page functions like a mini MLOps system, fully compliant with the assignment’s expectations for training, evaluation, and documentation.

Page 3 — EDA Gallery

Shows exploratory visualizations based on the ballistics dataset:

Distributions of range, muzzle velocity, group size

Correlations and pairwise plots

Conditional breakdowns (e.g., hit % by wind or range)

Scatterplots, histograms, heatmaps

Fully interactive via Streamlit

Page 4 — Dashboard

A polished, insight-focused dashboard summarizing:

Key ballistics KPIs

Hit probability trends

Environmental influence summaries

Bullet behavior visualizations

Interactive filters & cross-highlighting

Optional charts such as retained velocity vs BC, when relevant

This ties the EDA and ML pages together into actionable insights.

3. Machine Learning Models

The app includes two complementary machine-learning systems, each with different purposes.

Model A: Lightweight Hit-Probability Model (Pretrained)

Used on Page 1.

Logistic Regression trained offline

Base features:

Range

MV + SD

Wind speed

Group MOA

Target size MOA

DA, temperature, pressure, humidity

Fast inference

Suitable for demonstrations and intuitive teaching

Model Evaluation (Lightweight Logistic Regression Model)

The lightweight hit-probability model was evaluated on a held-out test set using the same semi-realistic dataset it was trained on. The following metrics summarize its performance:

Accuracy: 0.675

ROC AUC: 0.750

Log Loss: 0.5837

Confusion Matrix (rows = true, columns = predicted):

	Miss	Hit
Miss	50	18
Hit	21	31

Interpretation:
The model achieves moderate predictive performance consistent with expectations for a logistic regression trained on semi-realistic ballistic simulations.
Its ROC AUC of 0.75 indicates good discriminative ability for ranking high-probability vs low-probability shots.
Log Loss of 0.58 shows reasonable probability calibration without overconfident predictions.
This model is intended as a fast, educational approximation of real-world ballistic probability, not a firing solution.

Test Split: 80/20 on simulated dataset

Provides reliable trend patterns, though not personalized.

Model B: Full Calibrated Hit-Probability Model (User-Trained)

Used on Page 2.

Logistic Regression → StandardScaler → CalibratedClassifierCV

Includes interaction terms & nonlinear ballistics features

Supports user-uploaded real shooting data

Produces:

Calibrated probabilities

Train/test evaluation

Feature importances (SHAP)

Personalized predictions

Evaluation Metrics (computed dynamically):

Log Loss displayed in app

Automatically adjusts for small datasets

Uses stratified splits where possible

4. Limitations

Although designed with care, both models have important limitations:

Not a firing solution
These are statistical models, not physics solvers (no drag function, spin drift, Coriolis, etc.).

Simulated data
The lightweight model uses semi-realistic simulated data, not pressure-tested real-world firing data.

Dataset quality
User-provided data may have noise, inconsistencies, or insufficient samples.

No dynamic bullet drag modeling
Environmental factors are approximated; no G1/G7 drag curve integration.

UI feedback is educational only
Not intended for real-world ballistic decision-making or safety-critical use.