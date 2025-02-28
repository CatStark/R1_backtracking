import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from datetime import datetime
from scipy import stats

warnings.filterwarnings('ignore', category=UserWarning)


def get_model_choice():
    """Get user input for which models to run"""
    print("\nWhich models would you like to run?")
    print("1: Base (S = β₀ + β₁hB)")
    print("2: With Difficulty (S = β₀ + β₁hB + β₂Dir)")
    print("3: Full Model + Length Interaction (S = β₀ + β₁hB + β₂Dir + β₃hB×Dir + β₄Length + β₅hB×Length)")
    print("4: All logistic regression models")
    print("5: Length vs Backtracks linear regression (Backtracks = β₀ + β₁Length)")
    print("6: All models (logistic and linear)")

    while True:
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            if choice in ['1', '2', '3', '4', '5', '6']:
                return int(choice)
            print("Invalid choice. Please enter a number between 1 and 6.")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            exit()


def run_selected_models(df, model_choice):
    """Run the selected regression models following the formula structure"""
    results = {}

    # Create interaction terms
    df['HB_x_Diff'] = df['Has_Backtrack'] * df['difficulty_level']  # hB × Dir
    df['HB_x_Length'] = df['Has_Backtrack'] * df['Length']  # Has_Backtrack × Length interaction term

    logistic_model_specs = {
        1: (['Has_Backtrack'],
            'Model 1: Base (S = β₀ + β₁hB)'),

        2: (['Has_Backtrack', 'difficulty_level'],
            'Model 2: With Difficulty (S = β₀ + β₁hB + β₂Dir)'),

        3: (['Has_Backtrack', 'difficulty_level', 'HB_x_Diff', 'Length', 'HB_x_Length'],
            'Model 3: Full Model + Length Interaction (S = β₀ + β₁hB + β₂Dir + β₃hB×Dir + β₄Length + β₅hB×Length)')
    }

    # Define linear model for Length vs Backtracks
    linear_model_name = 'Model 4: Length vs Backtracks (Backtracks = β₀ + β₁Length)'

    if model_choice in [1, 2, 3]:  # Single logistic model
        vars, name = logistic_model_specs[model_choice]
        results[name] = run_logistic_model(df, vars, name)
    elif model_choice == 4:  # All logistic models
        for num, (vars, name) in logistic_model_specs.items():
            results[name] = run_logistic_model(df, vars, name)
    elif model_choice == 5:  # Only Length vs Backtracks linear model
        results[linear_model_name] = run_length_backtracks_model(df)
    else:  # All models (logistic and linear)
        for num, (vars, name) in logistic_model_specs.items():
            results[name] = run_logistic_model(df, vars, name)
        results[linear_model_name] = run_length_backtracks_model(df)

    return results


def run_length_backtracks_model(df):
    """Run a linear regression model for Length vs Number of Backtracks"""
    # We need a column with actual number of backtracks, not just binary Has_Backtrack
    # If not available, we can create a note about this
    backtracks_col = 'Num_Backtracks'

    # Create X (Length) and y (Num_Backtracks)
    X = df[['Length']].values
    y = df[backtracks_col].values if backtracks_col in df.columns else np.zeros(len(df))

    # Fit the model
    model = LinearRegression()
    model.fit(X, y)

    # Calculate statistics
    y_pred = model.predict(X)
    n = len(X)
    k = X.shape[1]
    residuals = y - y_pred
    sse = np.sum(residuals ** 2)
    mse = sse / (n - k - 1)
    se = np.sqrt(mse * np.linalg.inv(X.T @ X).diagonal())
    t_stats = model.coef_ / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))

    # Calculate R-squared and adjusted R-squared
    y_mean = np.mean(y)
    sst = np.sum((y - y_mean) ** 2)
    r_squared = 1 - (sse / sst)
    adjusted_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - k - 1))

    # Create a table for results
    table = pd.DataFrame({
        'Coefficient': [model.intercept_, model.coef_[0]],
        'Std. Error': [np.sqrt(
            mse * np.linalg.inv(np.column_stack((np.ones(n), X)).T @ np.column_stack((np.ones(n), X))).diagonal()[0]),
                       se[0]],
        't-value': [model.intercept_ / np.sqrt(
            mse * np.linalg.inv(np.column_stack((np.ones(n), X)).T @ np.column_stack((np.ones(n), X))).diagonal()[0]),
                    t_stats[0]],
        'p-value': [2 * (1 - stats.t.cdf(np.abs(model.intercept_ / np.sqrt(
            mse * np.linalg.inv(np.column_stack((np.ones(n), X)).T @ np.column_stack((np.ones(n), X))).diagonal()[0])),
                                         n - k - 1)), p_values[0]]
    }, index=['Intercept', 'Length'])

    # Format table
    for col in table.columns:
        table[col] = table[col].map('{:.4f}'.format)

    table[''] = ''
    # Add significance stars
    table.loc[table['p-value'].astype(float) < 0.01, ''] = '***'
    table.loc[(table['p-value'].astype(float) < 0.05) & (table['p-value'].astype(float) >= 0.01), ''] = '**'
    table.loc[(table['p-value'].astype(float) < 0.10) & (table['p-value'].astype(float) >= 0.05), ''] = '*'

    # Add R-squared information
    r2_row = pd.DataFrame([[''] * len(table.columns)],
                          columns=table.columns,
                          index=[f"R² = {r_squared:.4f}, Adjusted R² = {adjusted_r_squared:.4f}"])
    table = pd.concat([table, r2_row])

    return {
        'stats': {
            'coefficients': np.array([model.intercept_, model.coef_[0]]),
            'std_errors': np.array([np.sqrt(
                mse * np.linalg.inv(np.column_stack((np.ones(n), X)).T @ np.column_stack((np.ones(n), X))).diagonal()[
                    0]), se[0]]),
            't_stats': np.array([model.intercept_ / np.sqrt(
                mse * np.linalg.inv(np.column_stack((np.ones(n), X)).T @ np.column_stack((np.ones(n), X))).diagonal()[
                    0]), t_stats[0]]),
            'p_values': np.array([2 * (1 - stats.t.cdf(np.abs(model.intercept_ / np.sqrt(
                mse * np.linalg.inv(np.column_stack((np.ones(n), X)).T @ np.column_stack((np.ones(n), X))).diagonal()[
                    0])), n - k - 1)), p_values[0]]),
            'r_squared': r_squared,
            'adjusted_r_squared': adjusted_r_squared,
            'model_type': 'linear'
        },
        'variables': ['Length'],
        'table': table,
        'model': model
    }


def calculate_pseudo_r2(model, X, y):
    """Calculate McFadden's pseudo R-squared for logistic regression"""
    null_model = LogisticRegression(fit_intercept=True)
    null_model.fit(np.zeros((len(X), 1)), y)
    ll_null = null_model.score(np.zeros((len(X), 1)), y) * len(X)
    ll_model = model.score(X, y) * len(X)
    return 1 - (ll_model / ll_null)


def calculate_regression_stats(model, X, y):
    """Calculate standard regression statistics for logistic regression"""
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    predictions = model.predict_proba(X)[:, 1]
    V = np.diagflat(predictions * (1 - predictions))

    try:
        cov_matrix = np.linalg.inv(X_with_intercept.T @ V @ X_with_intercept)
        std_errors = np.sqrt(np.diag(cov_matrix))
    except np.linalg.LinAlgError:
        std_errors = np.array([np.nan] * (X.shape[1] + 1))

    all_coef = np.concatenate([model.intercept_, model.coef_[0]])
    z_scores = all_coef / std_errors
    p_values = 2 * (1 - stats.norm.cdf(abs(z_scores)))
    ci_lower = all_coef - 1.96 * std_errors
    ci_upper = all_coef + 1.96 * std_errors
    odds_ratios = np.exp(all_coef)

    return {
        'coefficients': all_coef,
        'std_errors': std_errors,
        'z_scores': z_scores,
        'p_values': p_values,
        'odds_ratios': odds_ratios,
        'odds_ratio_ci_lower': np.exp(ci_lower),
        'odds_ratio_ci_upper': np.exp(ci_upper),
        'pseudo_r2': calculate_pseudo_r2(model, X, y)
    }


def create_regression_table(stats_dict, variable_names):
    """Create a formatted regression table for text output"""
    table = pd.DataFrame({
        'Coefficient': stats_dict['coefficients'],
        'Std. Error': stats_dict['std_errors'],
        'z-value': stats_dict['z_scores'],
        'p-value': stats_dict['p_values'],
        'OR': stats_dict['odds_ratios'],
        'OR 95% CI Lower': stats_dict['odds_ratio_ci_lower'],
        'OR 95% CI Upper': stats_dict['odds_ratio_ci_upper']
    }, index=['Intercept'] + variable_names)

    for col in table.columns:
        table[col] = table[col].map('{:.4f}'.format)

    table[''] = ''
    # Updated significance levels
    table.loc[table['p-value'].astype(float) < 0.01, ''] = '***'
    table.loc[(table['p-value'].astype(float) < 0.05) & (table['p-value'].astype(float) >= 0.01), ''] = '**'
    table.loc[(table['p-value'].astype(float) < 0.10) & (table['p-value'].astype(float) >= 0.05), ''] = '*'

    pseudo_r2_row = pd.DataFrame([[''] * len(table.columns)],
                                 columns=table.columns,
                                 index=[f"Pseudo R² = {stats_dict['pseudo_r2']:.4f}"])
    table = pd.concat([table, pseudo_r2_row])

    return table


def create_pivoted_dataframe(all_results):
    """Create a pivoted DataFrame for CSV output"""
    # Get all model types
    has_linear = any(result.get('stats', {}).get('model_type') == 'linear' for result in all_results.values())

    # Initialize variables for all possible models
    variables = ['Intercept', 'Has_Backtrack', 'difficulty_level', 'HB_x_Diff', 'Length', 'HB_x_Length', 'Pseudo_R2']

    # Add R² for linear models if needed
    if has_linear:
        variables.append('R_squared')
        variables.append('Adjusted_R_squared')

    # Create clean model names
    models = list(all_results.keys())
    clean_models = [model_name.replace(' ', '_') for model_name in models]

    # For logistic models, use these metrics
    logistic_metrics = ['Coefficient', 'Std_Error', 'Z_value', 'P_value', 'OR',
                        'OR_95_CI_Lower', 'OR_95_CI_Upper', 'Significance']

    # For linear models, use these metrics
    linear_metrics = ['Coefficient', 'Std_Error', 'T_value', 'P_value', 'Significance']

    # Combine all possible metrics
    all_metrics = list(set(logistic_metrics + linear_metrics))

    # Initialize multi-index DataFrame
    index = pd.MultiIndex.from_product([variables, all_metrics],
                                       names=['Variable', 'Metric'])
    pivoted_df = pd.DataFrame(index=index, columns=clean_models)

    # Fill the DataFrame with values
    for model_name, model_data in all_results.items():
        clean_model_name = model_name.replace(' ', '_')
        stats_dict = model_data['stats']
        vars_list = ['Intercept'] + model_data['variables']

        # Check model type
        is_linear = stats_dict.get('model_type') == 'linear'

        # Fill statistics for each variable
        for i, var in enumerate(vars_list):
            if i < len(stats_dict['coefficients']):  # Make sure we don't go out of bounds
                pivoted_df.loc[(var, 'Coefficient'), clean_model_name] = stats_dict['coefficients'][i]
                pivoted_df.loc[(var, 'Std_Error'), clean_model_name] = stats_dict['std_errors'][i]

                # Different column names for different model types
                if is_linear:
                    pivoted_df.loc[(var, 'T_value'), clean_model_name] = stats_dict['t_stats'][i]
                else:
                    pivoted_df.loc[(var, 'Z_value'), clean_model_name] = stats_dict['z_scores'][i]

                pivoted_df.loc[(var, 'P_value'), clean_model_name] = stats_dict['p_values'][i]

                # Only logistic models have odds ratios
                if not is_linear:
                    pivoted_df.loc[(var, 'OR'), clean_model_name] = stats_dict['odds_ratios'][i]
                    pivoted_df.loc[(var, 'OR_95_CI_Lower'), clean_model_name] = stats_dict['odds_ratio_ci_lower'][i]
                    pivoted_df.loc[(var, 'OR_95_CI_Upper'), clean_model_name] = stats_dict['odds_ratio_ci_upper'][i]

                # Add significance stars - updated significance levels
                p_value = stats_dict['p_values'][i]
                if p_value < 0.01:
                    sig = '***'
                elif p_value < 0.05:
                    sig = '**'
                elif p_value < 0.10:
                    sig = '*'
                else:
                    sig = ''
                pivoted_df.loc[(var, 'Significance'), clean_model_name] = sig

        # Add R² information
        if is_linear:
            pivoted_df.loc[('R_squared', 'Coefficient'), clean_model_name] = stats_dict['r_squared']
            pivoted_df.loc[('Adjusted_R_squared', 'Coefficient'), clean_model_name] = stats_dict['adjusted_r_squared']
        else:
            # Add Pseudo R²
            pivoted_df.loc[('Pseudo_R2', 'Coefficient'), clean_model_name] = stats_dict['pseudo_r2']

    return pivoted_df


def run_logistic_model(df, variables, model_name):
    """Run a single logistic regression model"""
    X = df[variables].values
    y = df['Success'].values
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    stats = calculate_regression_stats(model, X, y)
    return {
        'stats': stats,
        'variables': variables,
        'table': create_regression_table(stats, variables)
    }
def run_model(df, variables, model_name):
    """Run a single logistic regression model"""
    X = df[variables].values
    y = df['Success'].values
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    stats = calculate_regression_stats(model, X, y)
    return {
        'stats': stats,
        'variables': variables,
        'table': create_regression_table(stats, variables)
    }

def save_txt_output(results, output_file, df):  # Added df parameter
    """Save results in text format"""
    with open(output_file, 'w') as f:
        f.write(f"Number of observations: {len(df)}\n\n")

        f.write("Multiple Logistic Regression Models\n")
        f.write("=================================\n\n")

        for model_name, model_results in results.items():
            f.write(f"{model_name}\n")
            f.write("-" * 50 + "\n")
            f.write(model_results['table'].to_string())
            f.write("\n\n")

        f.write("\nSignificance codes: 0 '***' 0.01 '**' 0.05 '*' 0.10\n")

def main():
    # Setup paths
    current_dir = Path(__file__).parent
    experiments_path = current_dir / 'experiments'
    latest_experiment = max([f for f in experiments_path.iterdir() if f.is_dir()],
                            key=lambda x: x.name)
    print(f"Processing experiment: {latest_experiment.name}")

    # Create output directory
    output_dir = latest_experiment / 'linear_regs'
    output_dir.mkdir(exist_ok=True)

    # Read data
    df = pd.read_csv(latest_experiment / 'results' / 'main_table.csv')

    # Print column names to verify data
    print("\nAvailable columns:")
    print(df.columns)

    # Get user's model choice
    model_choice = get_model_choice()

    # Run selected models
    results = run_selected_models(df, model_choice)

    # Create timestamp for file names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save text output
    txt_output = output_dir / f'multiple_models_results_{timestamp}.txt'
    save_txt_output(results, txt_output, df)

    # Create and save pivoted CSV
    pivoted_df = create_pivoted_dataframe(results)
    csv_output = output_dir / f'regression_results_{timestamp}.csv'
    pivoted_df.to_csv(csv_output)

    print(f"\nResults have been saved to:")
    print(f"Text file: {txt_output}")
    print(f"CSV file: {csv_output}")
    print("\nSignificance codes:")
    print("*** p < 0.01")
    print("**  p < 0.05")
    print("*   p < 0.10")


if __name__ == "__main__":
    main()