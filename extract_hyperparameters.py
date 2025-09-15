#!/usr/bin/env python3
"""
Hyperparameter Extraction Script for Method 6 Results

This script extracts all hyperparameters from the method6 log file and saves them
in a pandas-friendly CSV format. Each row represents one combination of classifier 
and regressor with all their respective hyperparameters.

Structure per combination:
- stint_classifier: Number of stints classifier
- compound_clf_1: Stint 1 compound classifier  
- stint_len_reg_1: Stint 1 length regressor
- compound_clf_2: Stint 2 compound classifier
- stint_len_reg_2: Stint 2 length regressor
- compound_clf_3: Stint 3 compound classifier
- stint_len_reg_3: Stint 3 length regressor
"""

import re
import pandas as pd
import ast
from pathlib import Path

def parse_best_params(params_str):
    """Parse the Best params string into a dictionary."""
    try:
        # Remove 'Best params: ' prefix and parse as Python dict
        params_dict_str = params_str.replace('Best params: ', '')
        return ast.literal_eval(params_dict_str)
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing params: {params_str}")
        print(f"Error: {e}")
        return {}

def extract_hyperparameters(log_file_path):
    """
    Extract all hyperparameters from the method6 log file.
    
    Returns:
        pandas.DataFrame: Each row contains all hyperparameters for one combination
    """
    
    # Read the log file
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split content by combinations
    combination_pattern = r'COMBINATION (\d+)/64: (.+?) \+ (.+?)$'
    combinations = re.findall(combination_pattern, content, re.MULTILINE)
    
    results = []
    
    # Process each combination
    for combo_num, classifier_type, regressor_type in combinations:
        print(f"Processing Combination {combo_num}: {classifier_type} + {regressor_type}")
        
        # Find the section for this combination
        start_pattern = f"COMBINATION {combo_num}/64:"
        end_pattern = f"COMBINATION {int(combo_num)+1}/64:" if int(combo_num) < 64 else "TESTING COMPLETED"
        
        start_idx = content.find(start_pattern)
        end_idx = content.find(end_pattern, start_idx)
        if end_idx == -1:
            end_idx = len(content)
        
        combo_section = content[start_idx:end_idx]
        
        # Extract all "Best params:" lines from this combination
        best_params_pattern = r'Best params: ({.*?})'
        param_matches = re.findall(best_params_pattern, combo_section)
        
        if len(param_matches) < 7:
            print(f"Warning: Expected 7 parameter sets for combination {combo_num}, found {len(param_matches)}")
            continue
        
        # Parse parameters for each model in the expected order
        try:
            stint_classifier_params = parse_best_params(f"Best params: {param_matches[0]}")
            compound_clf_1_params = parse_best_params(f"Best params: {param_matches[1]}")
            stint_len_reg_1_params = parse_best_params(f"Best params: {param_matches[2]}")
            compound_clf_2_params = parse_best_params(f"Best params: {param_matches[3]}")
            stint_len_reg_2_params = parse_best_params(f"Best params: {param_matches[4]}")
            compound_clf_3_params = parse_best_params(f"Best params: {param_matches[5]}")
            stint_len_reg_3_params = parse_best_params(f"Best params: {param_matches[6]}")
        except IndexError:
            print(f"Error: Could not extract all parameters for combination {combo_num}")
            continue
        
        # Create row dictionary
        row = {
            'combination_number': int(combo_num),
            'classifier_type': classifier_type,
            'regressor_type': regressor_type,
        }
        
        # Add stint classifier parameters
        for key, value in stint_classifier_params.items():
            clean_key = key.replace('classifier__', '')
            row[f'stint_classifier_{clean_key}'] = value
        
        # Add compound classifier 1 parameters
        for key, value in compound_clf_1_params.items():
            clean_key = key.replace('classifier__', '')
            row[f'compound_clf_1_{clean_key}'] = value
        
        # Add stint length regressor 1 parameters
        for key, value in stint_len_reg_1_params.items():
            clean_key = key.replace('regressor__', '')
            row[f'stint_len_reg_1_{clean_key}'] = value
        
        # Add compound classifier 2 parameters
        for key, value in compound_clf_2_params.items():
            clean_key = key.replace('classifier__', '')
            row[f'compound_clf_2_{clean_key}'] = value
        
        # Add stint length regressor 2 parameters
        for key, value in stint_len_reg_2_params.items():
            clean_key = key.replace('regressor__', '')
            row[f'stint_len_reg_2_{clean_key}'] = value
        
        # Add compound classifier 3 parameters
        for key, value in compound_clf_3_params.items():
            clean_key = key.replace('classifier__', '')
            row[f'compound_clf_3_{clean_key}'] = value
        
        # Add stint length regressor 3 parameters
        for key, value in stint_len_reg_3_params.items():
            clean_key = key.replace('regressor__', '')
            row[f'stint_len_reg_3_{clean_key}'] = value
        
        results.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Sort by combination number
    df = df.sort_values('combination_number').reset_index(drop=True)
    
    return df

def main():
    # Define file paths
    log_file = "method6_results_20250912_200057.log"
    output_file = "extracted_hyperparameters.csv"
    
    # Check if log file exists
    if not Path(log_file).exists():
        print(f"Error: Log file '{log_file}' not found in current directory")
        return
    
    print(f"Extracting hyperparameters from {log_file}...")
    
    # Extract hyperparameters
    df = extract_hyperparameters(log_file)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"Successfully extracted hyperparameters for {len(df)} combinations")
    print(f"Results saved to: {output_file}")
    
    # Print summary
    print("\nSummary:")
    print(f"Total combinations processed: {len(df)}")
    print(f"Columns in output file: {len(df.columns)}")
    print(f"First few column names: {list(df.columns[:10])}")
    
    # Show sample of data
    print("\nFirst few rows preview:")
    print(df[['combination_number', 'classifier_type', 'regressor_type']].head())
    
    # Show unique parameter names for each model type
    print("\nUnique parameters found:")
    all_cols = df.columns.tolist()
    
    stint_clf_cols = [col for col in all_cols if col.startswith('stint_classifier_')]
    print(f"Stint classifier parameters: {[col.replace('stint_classifier_', '') for col in stint_clf_cols]}")
    
    compound_clf_cols = [col for col in all_cols if col.startswith('compound_clf_1_')]
    print(f"Compound classifier parameters: {[col.replace('compound_clf_1_', '') for col in compound_clf_cols]}")
    
    stint_reg_cols = [col for col in all_cols if col.startswith('stint_len_reg_1_')]
    print(f"Stint length regressor parameters: {[col.replace('stint_len_reg_1_', '') for col in stint_reg_cols]}")

if __name__ == "__main__":
    main()
