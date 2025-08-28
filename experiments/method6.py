""" 6. Strategy-Segmented Chained Models

This approach builds on the \"Chained Predictions\" (Method 2) by first segmenting the data based on the number of stops and then training a chained model for each segment.

-   **Modeling:**
    1.  **Data Segmentation:** Split the dataset into subsets based on the number of stops in the strategy (e.g., one-stop races, two-stop races).
    2.  **Chained Models per Segment:** For each segment, train a separate chained prediction model as described in Method 2.
        -   The model for two-stop strategies would have a chain for Stint 1 and Stint 2.
        -   The model for three-stop strategies would have a chain for Stint 1, Stint 2, and Stint 3.
-   **Pros:**
    -   Creates more specialized models that can capture the distinct patterns of different stop strategies.
    -   Prevents aggressive, multi-stop strategies (e.g., starting on `SOFT`) from unduly influencing the models for conservative one-stop strategies.
-   **Cons:**
    -   Requires a pre-classification step to determine the number of stops, which can introduce errors.
    -   Reduces the amount of data available for training each individual model.
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
import xgboost as xgb
import catboost as cb

# warnings.filterwarnings('ignore')

# --- Helper for colored output ---
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# --- 0. Model Selection ---
# Choose from 'RandomForest', 'XGBoost', 'CatBoost'
SELECTED_MODEL = 'RandomForest'

models = {
    'RandomForest': {
        'classifier': RandomForestClassifier(random_state=42),
        'regressor': RandomForestRegressor(random_state=42)
    },
    'XGBoost': {
        'classifier': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
        'regressor': xgb.XGBRegressor(random_state=42)
    },
    'CatBoost': {
        'classifier': cb.CatBoostClassifier(random_state=42, verbose=0),
        'regressor': cb.CatBoostRegressor(random_state=42, verbose=0)
    }
}

print(f"{Colors.HEADER}--- Using {SELECTED_MODEL} models ---{Colors.ENDC}")
classifier = models[SELECTED_MODEL]['classifier']
regressor = models[SELECTED_MODEL]['regressor']


# --- 1. Load and Preprocess Data ---
print(f"{Colors.OKBLUE}--- Loading and Preprocessing Data ---{Colors.ENDC}")
races = pd.read_csv("data\\stints_2019-2024.csv")
df = races.copy()
df.dropna(inplace=True)
df['CompoundStrategy'] = df['CompoundStrategy'].apply(ast.literal_eval)
df['StintLengthStrategy'] = df['StintLengthStrategy'].apply(ast.literal_eval)
df = df[df['NumberOfStints'].isin([2, 3])]

for i in range(3):
    stint_num = i + 1
    df[f'Stint{stint_num}_Compound'] = df['CompoundStrategy'].apply(lambda x: x[i] if len(x) > i else None)
    df[f'Stint{stint_num}_Length'] = df['StintLengthStrategy'].apply(lambda x: x[i] if len(x) > i else None)

# --- 2. Define Features and Split Data ---
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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y['NumberOfStints'])

# --- 3. Train Initial Classifier for Number of Stints ---
print(f"{Colors.OKBLUE}--- Training Model for Number of Stints ---{Colors.ENDC}")
base_preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)
stint_model = Pipeline(steps=[('preprocessor', base_preprocessor), ('classifier', classifier)])
y_train_stints = y_train['NumberOfStints']
le_stints = LabelEncoder().fit(y['NumberOfStints'])
if SELECTED_MODEL in ['XGBoost', 'CatBoost']:
    y_train_stints = le_stints.transform(y_train_stints)
stint_model.fit(X_train, y_train_stints)


# --- 4. Train Segmented Chained Models ---
chained_models = {}

# Combine X and y for easier filtering
train_df = pd.concat([X_train, y_train], axis=1)

for num_stints_segment in [2, 3]:
    print(f"{Colors.OKCYAN}--- Training Chained Models for {num_stints_segment}-Stint Strategies ---{Colors.ENDC}")
    segment_df = train_df[train_df['NumberOfStints'] == num_stints_segment]
    X_train_segment = segment_df[base_features]
    y_train_segment = segment_df[y.columns]

    models_for_segment = {}

    # --- Stint 1 Models ---
    y1_compound_le = LabelEncoder().fit(y_train_segment['Stint1_Compound'])
    y_train_stint1_compound = y1_compound_le.transform(y_train_segment['Stint1_Compound'])
    stint1_compound_model = Pipeline(steps=[('preprocessor', base_preprocessor), ('classifier', classifier)])
    stint1_compound_model.fit(X_train_segment, y_train_stint1_compound)

    stint1_length_model = Pipeline(steps=[('preprocessor', base_preprocessor), ('regressor', regressor)])
    stint1_length_model.fit(X_train_segment, y_train_segment['Stint1_Length'])

    models_for_segment['stint1_compound_model'] = stint1_compound_model
    models_for_segment['stint1_length_model'] = stint1_length_model
    models_for_segment['y1_compound_le'] = y1_compound_le

    # --- Stint 2 Models ---
    X_train_stint2 = X_train_segment.copy()
    X_train_stint2['Stint1_Compound'] = y_train_segment['Stint1_Compound']
    X_train_stint2['Stint1_Length'] = y_train_segment['Stint1_Length']

    stint2_cat_features = categorical_features + ['Stint1_Compound']
    stint2_num_features = numerical_features + ['Stint1_Length']
    stint2_preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', stint2_num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), stint2_cat_features)
        ],
        remainder='passthrough'
    )

    y2_compound_le = LabelEncoder().fit(y_train_segment['Stint2_Compound'])
    y_train_stint2_compound = y2_compound_le.transform(y_train_segment['Stint2_Compound'])
    stint2_compound_model = Pipeline(steps=[('preprocessor', stint2_preprocessor), ('classifier', classifier)])
    stint2_compound_model.fit(X_train_stint2, y_train_stint2_compound)

    stint2_length_model = Pipeline(steps=[('preprocessor', stint2_preprocessor), ('regressor', regressor)])
    stint2_length_model.fit(X_train_stint2, y_train_segment['Stint2_Length'])

    models_for_segment['stint2_compound_model'] = stint2_compound_model
    models_for_segment['stint2_length_model'] = stint2_length_model
    models_for_segment['y2_compound_le'] = y2_compound_le

    # --- Stint 3 Models (only for 3-stint segment) ---
    if num_stints_segment == 3:
        X_train_stint3 = X_train_stint2.copy()
        X_train_stint3['Stint2_Compound'] = y_train_segment['Stint2_Compound']
        X_train_stint3['Stint2_Length'] = y_train_segment['Stint2_Length']

        stint3_cat_features = categorical_features + ['Stint1_Compound', 'Stint2_Compound']
        stint3_num_features = numerical_features + ['Stint1_Length', 'Stint2_Length']
        stint3_preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', stint3_num_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), stint3_cat_features)
            ],
            remainder='passthrough'
        )

        y3_compound_le = LabelEncoder().fit(y_train_segment['Stint3_Compound'])
        y_train_stint3_compound = y3_compound_le.transform(y_train_segment['Stint3_Compound'])
        stint3_compound_model = Pipeline(steps=[('preprocessor', stint3_preprocessor), ('classifier', classifier)])
        stint3_compound_model.fit(X_train_stint3, y_train_stint3_compound)

        stint3_length_model = Pipeline(steps=[('preprocessor', stint3_preprocessor), ('regressor', regressor)])
        stint3_length_model.fit(X_train_stint3, y_train_segment['Stint3_Length'])

        models_for_segment['stint3_compound_model'] = stint3_compound_model
        models_for_segment['stint3_length_model'] = stint3_length_model
        models_for_segment['y3_compound_le'] = y3_compound_le

    chained_models[num_stints_segment] = models_for_segment

# --- 5. Prediction and Evaluation on Test Set ---

def predict_strategy(X_row):
    """Predicts the full strategy for a single race instance."""
    # Predict number of stints to select the correct model chain
    pred = stint_model.predict(X_row)[0]
    num_stints = le_stints.inverse_transform([int(pred)])[0] if SELECTED_MODEL in ['XGBoost', 'CatBoost'] else pred

    # Select the appropriate chained model
    model_chain = chained_models[num_stints]
    
    # Predict Stint 1
    s1_compound_encoded = model_chain['stint1_compound_model'].predict(X_row)[0]
    s1_compound = model_chain['y1_compound_le'].inverse_transform([s1_compound_encoded])[0]
    s1_length = model_chain['stint1_length_model'].predict(X_row)[0]
    strategy = [(s1_compound, round(s1_length))]

    # Predict Stint 2
    X_row_s2 = X_row.copy()
    X_row_s2['Stint1_Compound'] = s1_compound
    X_row_s2['Stint1_Length'] = s1_length
    s2_compound_encoded = model_chain['stint2_compound_model'].predict(X_row_s2)[0]
    s2_compound = model_chain['y2_compound_le'].inverse_transform([s2_compound_encoded])[0]
    s2_length = model_chain['stint2_length_model'].predict(X_row_s2)[0]
    strategy.append((s2_compound, round(s2_length)))

    if num_stints > 2:
        # Predict Stint 3
        X_row_s3 = X_row_s2.copy()
        X_row_s3['Stint2_Compound'] = s2_compound
        X_row_s3['Stint2_Length'] = s2_length
        s3_compound_encoded = model_chain['stint3_compound_model'].predict(X_row_s3)[0]
        s3_compound = model_chain['y3_compound_le'].inverse_transform([s3_compound_encoded])[0]
        s3_length = model_chain['stint3_length_model'].predict(X_row_s3)[0]
        strategy.append((s3_compound, round(s3_length)))
        
    return num_stints, strategy

print(f"\n{Colors.HEADER}--- Evaluating on Test Set ---{Colors.ENDC}")
y_pred_stints = []
y_pred_strategies = []

for i in range(len(X_test)):
    row = X_test.iloc[[i]]
    pred_stints, pred_strategy = predict_strategy(row)
    y_pred_stints.append(pred_stints)
    y_pred_strategies.append(pred_strategy)

# Evaluate Number of Stints prediction
print(f"\n{Colors.BOLD}--- Number of Stints Classification Report ---{Colors.ENDC}")
print(classification_report(y_test['NumberOfStints'], y_pred_stints))

# Evaluate full strategy prediction
y_pred_strategies = pd.Series(y_pred_strategies, index=X_test.index)
for i in range(3):
    stint_num = i + 1
    true_compounds = y_test[f'Stint{stint_num}_Compound'].dropna()
    if true_compounds.empty:
        continue

    y_pred_stint_series = y_pred_strategies.loc[true_compounds.index]
    mask = y_pred_stint_series.apply(lambda s: len(s) > i)
    if not mask.any():
        continue

    true_compounds_eval = true_compounds[mask]
    y_pred_stint_eval = y_pred_stint_series[mask]
    pred_compounds = y_pred_stint_eval.apply(lambda s: s[i][0])
    true_lengths_eval = y_test.loc[true_compounds_eval.index, f'Stint{stint_num}_Length']
    pred_lengths = y_pred_stint_eval.apply(lambda s: s[i][1])

    print(f"{Colors.BOLD}--- Stint {stint_num} Compound Accuracy ---{Colors.ENDC}")
    print(f"Accuracy: {Colors.OKGREEN}{accuracy_score(true_compounds_eval, pred_compounds):.3f}{Colors.ENDC}")

    print(f"{Colors.BOLD}--- Stint {stint_num} Stint Length MAE ---{Colors.ENDC}")
    print(f"MAE: {Colors.OKGREEN}{mean_absolute_error(true_lengths_eval, pred_lengths):.3f}{Colors.ENDC} laps")

# --- 6. Example of a single prediction ---
print(f"\n{Colors.HEADER}--- Example Prediction ---{Colors.ENDC}")
example_index = 0
example_race = X_test.iloc[[example_index]]
true_stints = y_test.iloc[example_index]['NumberOfStints']

true_strategy_compounds = []
true_strategy_lengths = []
for i in range(1, int(true_stints) + 1):
    compound = y_test.iloc[example_index][f'Stint{i}_Compound']
    length = y_test.iloc[example_index][f'Stint{i}_Length']
    if pd.notna(compound):
        true_strategy_compounds.append(compound)
        true_strategy_lengths.append(round(length))

pred_stints, pred_strategy = predict_strategy(example_race)

print(f"{Colors.BOLD}Race Features:\n{Colors.ENDC}{example_race.to_string()}\n")
print(f"{Colors.OKBLUE}Predicted Number of Stints: {Colors.BOLD}{pred_stints}{Colors.ENDC}")
print(f"{Colors.OKBLUE}Predicted Strategy (Compound, Length): {Colors.BOLD}{pred_strategy}{Colors.ENDC}")
print(f"\n{Colors.OKGREEN}Actual Number of Stints: {Colors.BOLD}{true_stints}{Colors.ENDC}")
print(f"{Colors.OKGREEN}Actual Strategy (Compound, Length): {Colors.BOLD}{list(zip(true_strategy_compounds, true_strategy_lengths))}{Colors.ENDC}")
