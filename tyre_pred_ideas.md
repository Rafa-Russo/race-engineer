# Tyre Strategy Prediction: Modeling Approaches

## Introduction

The goal is to predict the tyre compound strategy and corresponding stint lengths for a race. The current approach of treating an entire strategy (e.g., `['HARD', 'MEDIUM']`) as a single class suffers from high dimensionality and class imbalance due to the large number of unique strategies.

This document outlines several alternative modeling approaches to address these challenges.

---

## 1. Multi-Label Classification for Compounds

This approach reframes the problem from a multi-class one to a multi-label one. Instead of predicting a single strategy, we predict a set of used compounds.

-   **Target Engineering:**
    -   The `CompoundStrategy` column would be transformed into a binary matrix where each column represents a compound (e.g., `SOFT`, `MEDIUM`, `HARD`).
    -   For each race, a `1` indicates a compound was used, and `0` otherwise.
-   **Modeling:**
    -   A multi-label classifier (e.g., `RandomForestClassifier` with multi-label support, or a one-vs-rest approach) would be trained to predict which compounds will be used in the race.
-   **Stint Length Prediction:**
    -   Separate regression models can be trained for each compound type to predict its stint length.
    -   These models would take the original features plus the predicted compounds as input.
-   **Pros:**
    -   Resolves the issue of having a massive number of classes.
    -   More robust to the class imbalance problem.
-   **Cons:**
    -   Does not preserve the order of the compounds in the strategy.
    -   The dependency between compound choices is not explicitly modeled.

---

## 2. Chained Predictions for Stints

This approach treats the strategy as a sequence and builds a chain of models to predict each stint one by one.

-   **Modeling:**
    1.  **Predict Number of Stints:** A classifier is first trained to predict the total number of stints (e.g., 1, 2, 3 stops).
    2.  **Stint 1 Model:** A model predicts the compound and stint length for the first stint using the initial race features.
    3.  **Stint 2 Model:** A second model predicts the compound and stint length for the second stint. Its inputs would include the initial race features **and** the predicted output from the Stint 1 model.
    4.  This chain continues for the predicted number of stints.
-   **Pros:**
    -   Captures the sequential, dependent nature of a race strategy.
    -   Models the strategy in a more realistic, step-by-step manner.
-   **Cons:**
    -   Complex to implement and manage multiple models.
    -   Errors from earlier models in the chain can propagate and negatively impact later predictions.

---

## 3. Stacked Ensemble Architecture

Stacking combines multiple models to improve predictive performance by using the predictions of base models as inputs to a higher-level meta-model. This directly addresses the issue of modeling the interplay between compound predictions.

-   **Architecture:**
    -   **Level 0 (Base Models):** Train several different models on the input data. These can be heterogeneous.
        -   *Example:* A classifier for each compound type (`is_soft_used`, `is_medium_used`, etc.), a regressor for the first stint length, a classifier for the number of stops.
    -   **Level 1 (Meta-Model):** The predictions from all Level 0 models are used as features to train a final meta-model. This model learns the relationships between the base predictions to make a more accurate final decision on the overall strategy.
-   **Pros:**
    -   Effectively models the interactions between different parts of the prediction (e.g., using SOFT tyres might influence the probability of using HARD tyres).
    -   Often achieves higher accuracy than individual models.
-   **Cons:**
    -   Can be computationally expensive.
    -   Increased complexity in implementation and tuning.

---

## 4. Sequence-to-Sequence Models (RNN/LSTM)

For a more advanced approach, Recurrent Neural Networks (RNNs), particularly LSTMs, can be used as they are specifically designed for sequence data.

-   **Modeling:**
    -   An RNN/LSTM model would be trained to generate a sequence of `(Compound, StintLength)` pairs.
-   **Input/Output:**
    -   **Input:** The initial race features.
    -   **Output:** A variable-length sequence representing the entire tyre strategy.
-   **Pros:**
    -   Theoretically the most powerful approach for sequential data.
    -   Can capture complex and long-term dependencies within the strategy.
-   **Cons:**
    -   Requires a larger dataset for effective training.
    -   Significantly more complex to build, train, and interpret than other methods.
    -   May be overkill if simpler models provide sufficient performance.
