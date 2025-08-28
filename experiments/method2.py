"""
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
"""


import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error
import warnings

# warnings.filterwarnings('ignore')

# --- 1. Load and Preprocess Data ---

# Load data
races = pd.read_csv("data\\stints_2019-2024.csv")
df = races.copy()

# Drop rows with missing values for simplicity
df.dropna(inplace=True)

# Convert string representation of lists to actual lists
df['CompoundStrategy'] = df['CompoundStrategy'].apply(ast.literal_eval)
df['StintLengthStrategy'] = df['StintLengthStrategy'].apply(ast.literal_eval)

# Filter for races with 2 or 3 stints (1 or 2 stops)
df = df[df['NumberOfStints'].isin([2, 3])]

# Create target columns for each stint
for i in range(3):
    stint_num = i + 1
    df[f'Stint{stint_num}_Compound'] = df['CompoundStrategy'].apply(lambda x: x[i] if len(x) > i else None)
    df[f'Stint{stint_num}_Length'] = df['StintLengthStrategy'].apply(lambda x: x[i] if len(x) > i else None)

# --- 2. Define Features and Split Data ---

# Base features for all models
base_features = [
    'Year', 'Circuit', 'Team', 'StartingPosition', 'AirTemp', 'Humidity',
    'Pressure', 'Rainfall', 'TrackTemp', 'WindDirection', 'WindSpeed',
    'mean_brake_time', 'std_brake_time', 'AvgSpeed', 'StdSpeed',
    'AvgSpeedDelta', 'StdSpeedDelta', 'AvgGearChanges'
]
categorical_features = ['Circuit', 'Team']
numerical_features = [f for f in base_features if f not in categorical_features]

X = df[base_features]
y = df[[
    'NumberOfStints',
    'Stint1_Compound', 'Stint1_Length',
    'Stint2_Compound', 'Stint2_Length',
    'Stint3_Compound', 'Stint3_Length'
]]

# Split data into training and test sets (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y['NumberOfStints'])

# --- 3. Train Chained Models ---

# Preprocessor for base features
base_preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# --- Model 0: Number of Stints ---
print("--- Training Model for Number of Stints ---")
stint_model = Pipeline(steps=[('preprocessor', base_preprocessor),
                              ('classifier', RandomForestClassifier(random_state=42))])
stint_model.fit(X_train, y_train['NumberOfStints'])

# --- Stint 1 Models ---
print("--- Training Stint 1 Models ---")
# Compound
y1_compound_le = LabelEncoder()
y_train_stint1_compound = y1_compound_le.fit_transform(y_train['Stint1_Compound'])
stint1_compound_model = Pipeline(steps=[('preprocessor', base_preprocessor),
                                        ('classifier', RandomForestClassifier(random_state=42))])
stint1_compound_model.fit(X_train, np.asarray(y_train_stint1_compound, dtype=int))

# Length
stint1_length_model = Pipeline(steps=[('preprocessor', base_preprocessor),
                                      ('regressor', RandomForestRegressor(random_state=42))])
stint1_length_model.fit(X_train, y_train['Stint1_Length'])

# --- Stint 2 Models ---
print("--- Training Stint 2 Models ---")
# Features for Stint 2 include Stint 1's output
X_train_stint2 = X_train.copy()
X_train_stint2['Stint1_Compound'] = y_train['Stint1_Compound']
X_train_stint2['Stint1_Length'] = y_train['Stint1_Length']

# Update preprocessor to handle the new compound feature
stint2_cat_features = categorical_features + ['Stint1_Compound']
stint2_num_features = numerical_features + ['Stint1_Length']
stint2_preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', stint2_num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), stint2_cat_features)
    ],
    remainder='passthrough'
)

# Compound
y2_compound_le = LabelEncoder()
y_train_stint2_compound = y2_compound_le.fit_transform(y_train['Stint2_Compound'])
stint2_compound_model = Pipeline(steps=[('preprocessor', stint2_preprocessor),
                                        ('classifier', RandomForestClassifier(random_state=42))])
stint2_compound_model.fit(X_train_stint2, np.asarray(y_train_stint2_compound, dtype=int))

# Length
stint2_length_model = Pipeline(steps=[('preprocessor', stint2_preprocessor),
                                      ('regressor', RandomForestRegressor(random_state=42))])
stint2_length_model.fit(X_train_stint2, y_train['Stint2_Length'])

# --- Stint 3 Models (only for 3-stint races) ---
print("--- Training Stint 3 Models ---")
df_3stints_train = pd.concat([X_train, y_train], axis=1).dropna(subset=['Stint3_Compound'])
X_train_stint3 = df_3stints_train[base_features].copy()
X_train_stint3['Stint1_Compound'] = df_3stints_train['Stint1_Compound']
X_train_stint3['Stint1_Length'] = df_3stints_train['Stint1_Length']
X_train_stint3['Stint2_Compound'] = df_3stints_train['Stint2_Compound']
X_train_stint3['Stint2_Length'] = df_3stints_train['Stint2_Length']
y_train_stint3 = df_3stints_train[['Stint3_Compound', 'Stint3_Length']]

# Update preprocessor for Stint 3 features
stint3_cat_features = categorical_features + ['Stint1_Compound', 'Stint2_Compound']
stint3_num_features = numerical_features + ['Stint1_Length', 'Stint2_Length']
stint3_preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', stint3_num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), stint3_cat_features)
    ],
    remainder='passthrough'
)

# Compound
y3_compound_le = LabelEncoder()
y_train_stint3_compound = y3_compound_le.fit_transform(y_train_stint3['Stint3_Compound'])
stint3_compound_model = Pipeline(steps=[('preprocessor', stint3_preprocessor),
                                        ('classifier', RandomForestClassifier(random_state=42))])
stint3_compound_model.fit(X_train_stint3, np.asarray(y_train_stint3_compound, dtype=int))

# Length
stint3_length_model = Pipeline(steps=[('preprocessor', stint3_preprocessor),
                                      ('regressor', RandomForestRegressor(random_state=42))])
stint3_length_model.fit(X_train_stint3, y_train_stint3['Stint3_Length'])


# --- 4. Prediction and Evaluation on Test Set ---

def predict_strategy(X_row):
    """Predicts the full strategy for a single race instance."""
    # Predict number of stints
    num_stints = stint_model.predict(X_row)[0]

    # Predict Stint 1
    s1_compound_encoded = stint1_compound_model.predict(X_row)[0]
    s1_compound = y1_compound_le.inverse_transform([s1_compound_encoded])[0]
    s1_length = float(stint1_length_model.predict(X_row)[0])

    strategy = [(s1_compound, round(s1_length))]

    # Predict Stint 2
    X_row_s2 = X_row.copy()
    X_row_s2['Stint1_Compound'] = s1_compound
    X_row_s2['Stint1_Length'] = s1_length
    
    s2_compound_encoded = stint2_compound_model.predict(X_row_s2)[0]
    s2_compound = y2_compound_le.inverse_transform([s2_compound_encoded])[0]
    s2_length = float(stint2_length_model.predict(X_row_s2)[0])
    strategy.append((s2_compound, round(s2_length)))

    if num_stints > 2:
        # Predict Stint 3
        X_row_s3 = X_row_s2.copy()
        X_row_s3['Stint2_Compound'] = s2_compound
        X_row_s3['Stint2_Length'] = s2_length

        s3_compound_encoded = stint3_compound_model.predict(X_row_s3)[0]
        s3_compound = y3_compound_le.inverse_transform([s3_compound_encoded])[0]
        s3_length = float(stint3_length_model.predict(X_row_s3)[0])
        strategy.append((s3_compound, round(s3_length)))
        
    return num_stints, strategy

print("\n--- Evaluating on Test Set ---")
y_pred_stints = []
y_pred_strategies = []

for i in range(len(X_test)):
    row = X_test.iloc[[i]]
    pred_stints, pred_strategy = predict_strategy(row)
    y_pred_stints.append(pred_stints)
    y_pred_strategies.append(pred_strategy)

# Evaluate Number of Stints prediction
print("\n--- Number of Stints Classification Report ---")
print(classification_report(y_test['NumberOfStints'], y_pred_stints))

# Evaluate full strategy prediction (Compound and Lengths)
# This is a more complex evaluation, we'll check compound and length accuracy for each stint
y_pred_strategies = pd.Series(y_pred_strategies, index=X_test.index)
for i in range(3):
    stint_num = i + 1
    true_compounds = y_test[f'Stint{stint_num}_Compound'].dropna()
    if true_compounds.empty:
        continue

    # Align predictions with true values by index
    y_pred_stint_series = y_pred_strategies.loc[true_compounds.index]

    # Create a mask for predictions that are long enough to have this stint
    mask = y_pred_stint_series.apply(lambda s: len(s) > i)

    if not mask.any():
        print(f"\n--- Stint {stint_num} Compound Accuracy ---")
        print("Accuracy: N/A (no valid predictions)")
        print(f"\n--- Stint {stint_num} Stint Length MAE ---")
        print("MAE: N/A (no valid predictions)")
        continue

    # Filter both true and predicted values using the mask
    true_compounds_eval = true_compounds[mask]
    y_pred_stint_eval = y_pred_stint_series[mask]

    pred_compounds = y_pred_stint_eval.apply(lambda s: s[i][0])

    true_lengths_eval = y_test.loc[true_compounds_eval.index, f'Stint{stint_num}_Length']
    pred_lengths = y_pred_stint_eval.apply(lambda s: s[i][1])

    print(f"\n--- Stint {stint_num} Compound Accuracy ---")
    print(f"Accuracy: {accuracy_score(true_compounds_eval, pred_compounds):.3f}")

    print(f"\n--- Stint {stint_num} Stint Length MAE ---")
    print(f"MAE: {mean_absolute_error(true_lengths_eval, pred_lengths):.3f} laps")


# Example of a single prediction
print("\n--- Example Prediction ---")
example_index = 0
example_race = X_test.iloc[[example_index]]
true_stints = y_test.iloc[example_index]['NumberOfStints']

# Reconstruct the true strategy from individual stint columns
true_strategy_compounds = []
true_strategy_lengths = []
for i in range(1, true_stints + 1):
    compound = y_test.iloc[example_index][f'Stint{i}_Compound']
    length = y_test.iloc[example_index][f'Stint{i}_Length']
    if pd.notna(compound):
        true_strategy_compounds.append(compound)
        true_strategy_lengths.append(round(float(length)))

pred_stints, pred_strategy = predict_strategy(example_race)

print(f"Race Features:\n{example_race.to_string()}\n")
print(f"Predicted Number of Stints: {pred_stints}")
print(f"Predicted Strategy (Compound, Length): {pred_strategy}")
print(f"\nActual Number of Stints: {true_stints}")
print(f"Actual Strategy (Compound, Length): {list(zip(true_strategy_compounds, true_strategy_lengths))}")
