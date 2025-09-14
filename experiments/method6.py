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
import ast
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error
import xgboost as xgb
import catboost as cb
import logging
from datetime import datetime

# warnings.filterwarnings('ignore')

# --- Setup logging configuration ---
def setup_logger():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"method6_results_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger('method6_logger')
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Setup logger
logger = setup_logger()

def log_print(message, color_code=""):
    """Custom print function that logs to file and prints to console with colors"""
    # Strip color codes for file logging
    clean_message = message
    if color_code:
        # Remove ANSI color codes for clean file logging
        import re
        clean_message = re.sub(r'\033\[[0-9;]*m', '', str(message))
    
    logger.info(clean_message)
    print(f"{color_code}{message}{Colors.ENDC if color_code else ''}")

# Initialize results storage
all_results = []

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

# --- 0. Model and Hyperparameter Configuration ---
# We'll test all possible combinations
USE_GRID_SEARCH = True # Set to False to use default parameters

model_configs = {
    'RandomForest': {
        'classifier': RandomForestClassifier(random_state=42),
        'regressor': RandomForestRegressor(random_state=42),
        'param_grid': {
            'classifier__n_estimators': [50, 100, 300, 500],
            'classifier__max_depth': [3, 6, 10, 20, None],
            'regressor__n_estimators': [50, 100, 300, 500],
            'regressor__max_depth': [3, 6, 10, 20, None],
        }
    },
    'XGBoost': {
        'classifier': xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'),
        'regressor': xgb.XGBRegressor(random_state=42),
        'param_grid': {
            'classifier__n_estimators': [50, 100, 300, 500],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'classifier__max_depth': [3, 6, 10],
            'classifier__reg_lambda': [0, 1, 10],
            'classifier__reg_alpha': [0, 1, 10],
            'regressor__n_estimators': [50, 100, 300, 500],
            'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'regressor__max_depth': [3, 6, 10],
            'regressor__reg_lambda': [0, 1, 10],
            'regressor__reg_alpha': [0, 1, 10],
        }
    },
    'CatBoost': {
        'classifier': cb.CatBoostClassifier(random_state=42, verbose=0),
        'regressor': cb.CatBoostRegressor(random_state=42, verbose=0),
        'param_grid': {
            'classifier__iterations': [50, 100, 300, 500],
            'classifier__depth': [4, 6, 10],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__l2_leaf_reg': [1, 3, 5, 10],
            'regressor__iterations': [50, 100, 300, 500],
            'regressor__depth': [4, 6, 10],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__l2_leaf_reg': [1, 3, 5, 10],
        }
    },
    'SVM': {
        'classifier': SVC(random_state=42, probability=True),
        'regressor': SVR(),
        'param_grid': {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__gamma': ['scale', 'auto'],
            'classifier__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'regressor__C': [0.01, 0.1, 1, 10, 100],
            'regressor__epsilon': [0.1, 0.2, 0.5],
            'regressor__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        }
    },
    'Linear': {
        'classifier': LogisticRegression(random_state=42, max_iter=500),
        'regressor': LinearRegression(),
        'param_grid': {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l2'],
            'regressor__fit_intercept': [True, False],
        }
    },
    'Ridge': {
        'classifier': RidgeClassifier(random_state=42),
        'regressor': Ridge(random_state=42),
        'param_grid': {
            'classifier__alpha': [0.1, 1.0, 10.0],
            'regressor__alpha': [0.1, 1.0, 10.0],
        }
    },
    'GradientBoosting': {
        'classifier': GradientBoostingClassifier(random_state=42),
        'regressor': GradientBoostingRegressor(random_state=42),
        'param_grid': {
            'classifier__n_estimators': [50, 100, 300, 500],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__max_depth': [3, 6, 10],
            'regressor__n_estimators': [50, 100, 300, 500],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__max_depth': [3, 6, 10],
        }
    },
    'KNN': {
        'classifier': KNeighborsClassifier(),
        'regressor': KNeighborsRegressor(),
        'param_grid': {
            'classifier__n_neighbors': [3, 5, 7, 11, 15],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'regressor__n_neighbors': [3, 5, 7, 11, 15],
            'regressor__weights': ['uniform', 'distance'],
            'regressor__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        }
    }
}

# Get all model names for combinations
model_names = list(model_configs.keys())
total_combinations = len(model_names) ** 2

log_print("="*80, Colors.HEADER)
log_print("STRATEGY-SEGMENTED CHAINED MODELS - COMPREHENSIVE TESTING", Colors.HEADER)
log_print(f"Testing all combinations of {len(model_names)} classifiers and {len(model_names)} regressors", Colors.HEADER)
log_print(f"Total combinations to test: {total_combinations}", Colors.HEADER)
log_print(f"Model types: {model_names}", Colors.HEADER)
log_print("="*80, Colors.HEADER)

def run_combination(selected_classifier, selected_regressor):
    """Run the complete pipeline for a specific classifier-regressor combination"""
    
    log_print(f"\n{'='*60}", Colors.HEADER)
    log_print(f"TESTING COMBINATION: {selected_classifier} (Classifier) + {selected_regressor} (Regressor)", Colors.HEADER)
    log_print(f"{'='*60}", Colors.HEADER)
    
    # Get model configurations
    classifier_config = model_configs[selected_classifier]
    regressor_config = model_configs[selected_regressor]
    
    classifier = classifier_config['classifier']
    regressor = regressor_config['regressor']
    
    param_grid_clf = {k: v for k, v in classifier_config['param_grid'].items() if k.startswith('classifier')}
    param_grid_reg = {k: v for k, v in regressor_config['param_grid'].items() if k.startswith('regressor')}
    param_grid = {**param_grid_clf, **param_grid_reg}

    # --- 1. Load and Preprocess Data ---
    log_print("--- Loading and Preprocessing Data ---", Colors.OKBLUE)
    races = pd.read_csv("data\\stints_2019-2025.csv")
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
    log_print("--- Training Model for Number of Stints ---", Colors.OKBLUE)
    # Define master categories to ensure all encoders are consistent
    all_circuits = sorted(df['Circuit'].unique())
    all_teams = sorted(df['Team'].unique())
    all_compounds = sorted(list(set(c for strategy in df['CompoundStrategy'] for c in strategy)))

    def create_base_preprocessor(model_name):
        if model_name == 'SVM':
            # SVM requires feature scaling
            return ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, categories=[all_circuits, all_teams]), categorical_features)
                ],
                remainder='passthrough'
            )
        else:
            return ColumnTransformer(
                transformers=[
                    ('num', 'passthrough', numerical_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, categories=[all_circuits, all_teams]), categorical_features)
                ],
                remainder='passthrough'
            )

    def train_model_with_grid_search(pipeline, X_train, y_train, model_type):
        if USE_GRID_SEARCH:
            grid = {k: v for k, v in param_grid.items() if k.startswith(model_type)}
            if not grid:
                model = pipeline
            else:
                scoring = 'accuracy' if model_type == 'classifier' else 'neg_mean_absolute_error'
                log_print(f"--- Performing GridSearchCV for {model_type} ---")
                log_print(f"Hyperparameter grid being searched: {grid}")
                model = GridSearchCV(pipeline, grid, cv=3, n_jobs=-1, scoring=scoring)
        else:
            model = pipeline
        
        model.fit(X_train, y_train)
        
        if USE_GRID_SEARCH and grid:
            log_print(f"Best params: {model.best_params_}")
            
        return model

    stint_model_pipeline = Pipeline(steps=[('preprocessor', create_base_preprocessor(selected_classifier)), ('classifier', classifier)])
    y_train_stints = y_train['NumberOfStints']
    le_stints = LabelEncoder().fit(y['NumberOfStints'])
    if selected_classifier in ['XGBoost', 'CatBoost', 'SVM']:
        y_train_stints = le_stints.transform(y_train_stints)

    stint_model = train_model_with_grid_search(stint_model_pipeline, X_train, y_train_stints, 'classifier')

    # --- 4. Train Segmented Chained Models ---
    chained_models = {}

    # Combine X and y for easier filtering
    train_df = pd.concat([X_train, y_train], axis=1)

    for num_stints_segment in [2, 3]:
        log_print(f"--- Training Chained Models for {num_stints_segment}-Stint Strategies ---", Colors.OKCYAN)
        segment_df = train_df[train_df['NumberOfStints'] == num_stints_segment]
        X_train_segment = segment_df[base_features]
        y_train_segment = segment_df[y.columns]

        models_for_segment = {}

        # --- Stint 1 Models ---
        y1_compound_le = LabelEncoder().fit(y_train_segment['Stint1_Compound'])
        y_train_stint1_compound = y1_compound_le.transform(y_train_segment['Stint1_Compound'])
        stint1_compound_pipeline = Pipeline(steps=[('preprocessor', create_base_preprocessor(selected_classifier)), ('classifier', classifier)])
        stint1_compound_model = train_model_with_grid_search(stint1_compound_pipeline, X_train_segment, y_train_stint1_compound, 'classifier')

        stint1_length_pipeline = Pipeline(steps=[('preprocessor', create_base_preprocessor(selected_regressor)), ('regressor', regressor)])
        stint1_length_model = train_model_with_grid_search(stint1_length_pipeline, X_train_segment, y_train_segment['Stint1_Length'], 'regressor')

        models_for_segment['stint1_compound_model'] = stint1_compound_model
        models_for_segment['stint1_length_model'] = stint1_length_model
        models_for_segment['y1_compound_le'] = y1_compound_le

        # --- Stint 2 Models ---
        X_train_stint2 = X_train_segment.copy()
        X_train_stint2['Stint1_Compound'] = y_train_segment['Stint1_Compound']
        X_train_stint2['Stint1_Length'] = y_train_segment['Stint1_Length']

        stint2_cat_features = categorical_features + ['Stint1_Compound']
        stint2_num_features = numerical_features + ['Stint1_Length']
        
        def create_stint2_preprocessor(model_name):
            if model_name == 'SVM':
                return ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), stint2_num_features),
                        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, categories=[all_circuits, all_teams, all_compounds]), stint2_cat_features)
                    ],
                    remainder='passthrough'
                )
            else:
                return ColumnTransformer(
                    transformers=[
                        ('num', 'passthrough', stint2_num_features),
                        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, categories=[all_circuits, all_teams, all_compounds]), stint2_cat_features)
                    ],
                    remainder='passthrough'
                )
        
        stint2_compound_preprocessor = create_stint2_preprocessor(selected_classifier)
        stint2_length_preprocessor = create_stint2_preprocessor(selected_regressor)

        y2_compound_le = LabelEncoder().fit(y_train_segment['Stint2_Compound'])
        y_train_stint2_compound = y2_compound_le.transform(y_train_segment['Stint2_Compound'])
        stint2_compound_pipeline = Pipeline(steps=[('preprocessor', stint2_compound_preprocessor), ('classifier', classifier)])
        stint2_compound_model = train_model_with_grid_search(stint2_compound_pipeline, X_train_stint2, y_train_stint2_compound, 'classifier')

        stint2_length_pipeline = Pipeline(steps=[('preprocessor', stint2_length_preprocessor), ('regressor', regressor)])
        stint2_length_model = train_model_with_grid_search(stint2_length_pipeline, X_train_stint2, y_train_segment['Stint2_Length'], 'regressor')

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
            
            def create_stint3_preprocessor(model_name):
                if model_name == 'SVM':
                    return ColumnTransformer(
                        transformers=[
                            ('num', StandardScaler(), stint3_num_features),
                            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, categories=[all_circuits, all_teams, all_compounds, all_compounds]), stint3_cat_features)
                        ],
                        remainder='passthrough'
                    )
                else:
                    return ColumnTransformer(
                        transformers=[
                            ('num', 'passthrough', stint3_num_features),
                            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, categories=[all_circuits, all_teams, all_compounds, all_compounds]), stint3_cat_features)
                        ],
                        remainder='passthrough'
                    )
            
            stint3_compound_preprocessor = create_stint3_preprocessor(selected_classifier)
            stint3_length_preprocessor = create_stint3_preprocessor(selected_regressor)

            y3_compound_le = LabelEncoder().fit(y_train_segment['Stint3_Compound'])
            y_train_stint3_compound = y3_compound_le.transform(y_train_segment['Stint3_Compound'])
            stint3_compound_pipeline = Pipeline(steps=[('preprocessor', stint3_compound_preprocessor), ('classifier', classifier)])
            stint3_compound_model = train_model_with_grid_search(stint3_compound_pipeline, X_train_stint3, y_train_stint3_compound, 'classifier')

            stint3_length_pipeline = Pipeline(steps=[('preprocessor', stint3_length_preprocessor), ('regressor', regressor)])
            stint3_length_model = train_model_with_grid_search(stint3_length_pipeline, X_train_stint3, y_train_segment['Stint3_Length'], 'regressor')

            models_for_segment['stint3_compound_model'] = stint3_compound_model
            models_for_segment['stint3_length_model'] = stint3_length_model
            models_for_segment['y3_compound_le'] = y3_compound_le

        chained_models[num_stints_segment] = models_for_segment

    # --- 5. Prediction and Evaluation on Test Set ---

    def predict_strategy(X_row):
        """Predicts the full strategy for a single race instance."""
        # Predict number of stints to select the correct model chain
        pred = stint_model.predict(X_row)[0]
        num_stints = le_stints.inverse_transform([int(pred)])[0] if selected_classifier in ['XGBoost', 'CatBoost', 'SVM'] else pred

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

    log_print("\n--- Evaluating on Test Set ---", Colors.HEADER)
    y_pred_stints = []
    y_pred_strategies = []

    for i in range(len(X_test)):
        row = X_test.iloc[[i]]
        pred_stints, pred_strategy = predict_strategy(row)
        y_pred_stints.append(pred_stints)
        y_pred_strategies.append(pred_strategy)

    # Evaluate Number of Stints prediction
    log_print("\n--- Number of Stints Classification Report ---", Colors.BOLD)
    stint_report = classification_report(y_test['NumberOfStints'], y_pred_stints)
    log_print(stint_report)
    
    stint_accuracy = accuracy_score(y_test['NumberOfStints'], y_pred_stints)

    # Evaluate full strategy prediction
    y_pred_strategies = pd.Series(y_pred_strategies, index=X_test.index)
    stint_compound_accuracies = {}
    stint_length_maes = {}
    
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

        compound_accuracy = accuracy_score(true_compounds_eval, pred_compounds)
        length_mae = mean_absolute_error(true_lengths_eval, pred_lengths)
        
        stint_compound_accuracies[stint_num] = compound_accuracy
        stint_length_maes[stint_num] = length_mae

        log_print(f"--- Stint {stint_num} Compound Accuracy ---", Colors.BOLD)
        log_print(f"Accuracy: {compound_accuracy:.3f}", Colors.OKGREEN)

        log_print(f"--- Stint {stint_num} Stint Length MAE ---", Colors.BOLD)
        log_print(f"MAE: {length_mae:.3f} laps", Colors.OKGREEN)

    # --- 6. Example of a single prediction ---
    log_print("\n--- Example Prediction ---", Colors.HEADER)
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

    log_print(f"Race Features:\n{example_race.to_string()}\n", Colors.BOLD)
    log_print(f"Predicted Number of Stints: {pred_stints}", Colors.OKBLUE)
    log_print(f"Predicted Strategy (Compound, Length): {pred_strategy}", Colors.OKBLUE)
    log_print(f"\nActual Number of Stints: {true_stints}", Colors.OKGREEN)
    log_print(f"Actual Strategy (Compound, Length): {list(zip(true_strategy_compounds, true_strategy_lengths))}", Colors.OKGREEN)

    # Return results for this combination
    return {
        'classifier': selected_classifier,
        'regressor': selected_regressor,
        'stint_accuracy': stint_accuracy,
        'stint_compound_accuracies': stint_compound_accuracies,
        'stint_length_maes': stint_length_maes
    }

# --- Main execution: Test all combinations ---
if __name__ == "__main__":
    log_print("="*80, Colors.HEADER)
    log_print("STARTING COMPREHENSIVE TESTING", Colors.HEADER)
    log_print("="*80, Colors.HEADER)
    
    combination_count = 0
    total_combinations = len(model_names) ** 2
    
    try:
        # Test all combinations
        for classifier_name in model_names:
            for regressor_name in model_names:
                combination_count += 1
                
                log_print(f"\n{'='*80}", Colors.WARNING)
                log_print(f"COMBINATION {combination_count}/{total_combinations}: {classifier_name} + {regressor_name}", Colors.WARNING)
                log_print(f"{'='*80}", Colors.WARNING)
                
                try:
                    result = run_combination(classifier_name, regressor_name)
                    all_results.append(result)
                    
                    log_print(f"✓ COMPLETED: {classifier_name} + {regressor_name}", Colors.OKGREEN)
                    
                except Exception as e:
                    log_print(f"✗ FAILED: {classifier_name} + {regressor_name}", Colors.FAIL)
                    log_print(f"Error: {str(e)}", Colors.FAIL)
                    # Continue with next combination even if one fails
                    continue
        
        # --- Results Summary ---
        log_print("\n" + "="*80, Colors.HEADER)
        log_print("COMPREHENSIVE RESULTS SUMMARY", Colors.HEADER)
        log_print("="*80, Colors.HEADER)
        
        if all_results:
            # Create results DataFrame for easy analysis
            results_df = pd.DataFrame(all_results)
            
            # Summary statistics
            log_print(f"\nSuccessfully tested {len(all_results)} combinations out of {total_combinations} total combinations", Colors.OKGREEN)
            
            # Find best combinations for each metric
            best_stint_accuracy = results_df.loc[results_df['stint_accuracy'].idxmax()]
            log_print(f"\nBest Number of Stints Accuracy: {best_stint_accuracy['stint_accuracy']:.3f}", Colors.OKGREEN)
            log_print(f"  Classifier: {best_stint_accuracy['classifier']}, Regressor: {best_stint_accuracy['regressor']}", Colors.OKGREEN)
            
            # Best stint compound accuracies for each stint
            for stint_num in [1, 2, 3]:
                compound_col = 'stint_compound_accuracies'
                available_results = [r for r in all_results if stint_num in r[compound_col]]
                if available_results:
                    best_combo = max(available_results, key=lambda x: x[compound_col][stint_num])
                    log_print(f"\nBest Stint {stint_num} Compound Accuracy: {best_combo[compound_col][stint_num]:.3f}", Colors.OKGREEN)
                    log_print(f"  Classifier: {best_combo['classifier']}, Regressor: {best_combo['regressor']}", Colors.OKGREEN)
            
            # Best stint length MAEs for each stint (lower is better)
            for stint_num in [1, 2, 3]:
                mae_col = 'stint_length_maes'
                available_results = [r for r in all_results if stint_num in r[mae_col]]
                if available_results:
                    best_combo = min(available_results, key=lambda x: x[mae_col][stint_num])
                    log_print(f"\nBest Stint {stint_num} Length MAE: {best_combo[mae_col][stint_num]:.3f} laps", Colors.OKGREEN)
                    log_print(f"  Classifier: {best_combo['classifier']}, Regressor: {best_combo['regressor']}", Colors.OKGREEN)
            
            # Save detailed results to CSV
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_csv = f"method6_detailed_results_{timestamp}.csv"
            
            # Flatten the nested dictionaries for CSV export
            flattened_results = []
            for result in all_results:
                flat_result = {
                    'classifier': result['classifier'],
                    'regressor': result['regressor'],
                    'stint_accuracy': result['stint_accuracy']
                }
                
                # Add stint compound accuracies
                for stint_num, accuracy in result['stint_compound_accuracies'].items():
                    flat_result[f'stint{stint_num}_compound_accuracy'] = accuracy
                
                # Add stint length MAEs
                for stint_num, mae in result['stint_length_maes'].items():
                    flat_result[f'stint{stint_num}_length_mae'] = mae
                
                flattened_results.append(flat_result)
            
            results_df = pd.DataFrame(flattened_results)
            results_df.to_csv(results_csv, index=False)
            log_print(f"\nDetailed results saved to: {results_csv}", Colors.OKBLUE)
            
            # Display top 5 combinations by stint accuracy
            log_print("\nTOP 5 COMBINATIONS BY STINT ACCURACY:", Colors.BOLD)
            top_combinations = results_df.nlargest(5, 'stint_accuracy')[['classifier', 'regressor', 'stint_accuracy']]
            for idx, row in top_combinations.iterrows():
                log_print(f"  {row['classifier']} + {row['regressor']}: {row['stint_accuracy']:.3f}", Colors.OKCYAN)
        
        else:
            log_print("No combinations completed successfully!", Colors.FAIL)
            
    except KeyboardInterrupt:
        log_print(f"\nTesting interrupted by user after {combination_count} combinations", Colors.WARNING)
        log_print(f"Completed {len(all_results)} combinations successfully", Colors.WARNING)
    
    except Exception as e:
        log_print(f"Unexpected error occurred: {str(e)}", Colors.FAIL)
    
    finally:
        log_print("\nTesting session completed. Check log files for detailed results.", Colors.HEADER)
