## File largely written by GitHub Copilot running Claude Sonnet 4.5

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load and prepare dataset."""
    df = pd.read_csv(csv_path)
    return df


def experiment_1_logistic_regression(df: pd.DataFrame) -> dict:
    """
    Task 1: Logistic Regression - Modeling Success Probability
    Predict accuracy using coordination and activity metrics.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: Logistic Regression - Success Probability")
    print("="*80)
    
    # Core features
    core_features = ['num_subagents', 'subagent_success_avg', 'subagent_similarity']
    # Activity features
    activity_features = ['total_tool_calls', 'lead_agent_messages', 'subagent_messages', 
                        'total_errors', 'num_turns']
    
    all_features = core_features + activity_features
    
    # Prepare data (drop rows with NaN in core features)
    df_clean = df.dropna(subset=core_features + ['accuracy'])
    
    # If we don't have enough data, use all runs with imputation
    if len(df_clean) < 3:
        print("Warning: Using all runs with mean imputation for missing values")
        df_clean = df.copy()
        for col in core_features:
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
    
    X = df_clean[all_features]
    y = df_clean['accuracy']
    
    # Handle any remaining NaN
    X = X.fillna(0)
    
    # Check if we have both classes
    if len(y.unique()) < 2:
        print(f"\nWarning: Only one class present in data (all {y.iloc[0]})")
        print("Cannot fit logistic regression. Showing feature statistics instead.")
        
        # Show correlations with accuracy as proxy
        print("\nFeature Summary Statistics:")
        print(X.describe().T[['mean', 'std', 'min', 'max']])
        
        return {
            'model': None,
            'accuracy': None,
            'coefficients': None,
            'predictions': None,
            'probabilities': None,
            'features': all_features,
            'note': 'Insufficient class variation'
        }
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit logistic regression
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)
    
    # Predictions and metrics
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]
    
    accuracy = (y_pred == y).mean()
    
    # Feature importance
    coefficients = pd.DataFrame({
        'feature': all_features,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print(f"\nModel Accuracy: {accuracy:.2%}")
    print(f"\nTop Feature Coefficients:")
    print(coefficients.head(10).to_string(index=False))
    
    results = {
        'model': model,
        'accuracy': accuracy,
        'coefficients': coefficients,
        'predictions': y_pred,
        'probabilities': y_prob,
        'features': all_features
    }
    
    return results


def experiment_2_linear_regression(df: pd.DataFrame) -> dict:
    """
    Task 2: Multiple Linear Regression - Efficiency and Cost Trade-offs
    Predict cost_usd and time_seconds.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: Multiple Linear Regression - Cost & Time Analysis")
    print("="*80)
    
    predictors = ['num_subagents', 'total_tool_calls', 'num_turns', 
                 'total_errors', 'websearch_calls', 'webfetch_calls']
    
    results = {}
    
    for target in ['cost_usd', 'time_seconds']:
        print(f"\n--- Predicting {target} ---")
        
        # Prepare data
        df_clean = df.dropna(subset=predictors + [target])
        X = df_clean[predictors]
        y = df_clean[target]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Predictions and metrics
        y_pred = model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Feature importance
        coefficients = pd.DataFrame({
            'feature': predictors,
            'coefficient': model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
        
        print(f"R² Score: {r2:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"\nTop Predictors:")
        print(coefficients.head().to_string(index=False))
        
        results[target] = {
            'model': model,
            'r2': r2,
            'rmse': rmse,
            'coefficients': coefficients,
            'predictions': y_pred,
            'actual': y.values
        }
    
    return results


def experiment_3_correlation_analysis(df: pd.DataFrame, output_dir: str = 'experiments') -> dict:
    """
    Task 3: Correlation Analysis - Linking Subagent Behavior to Accuracy
    Generate correlation heatmap.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 3: Correlation Analysis")
    print("="*80)
    
    # Structural variables
    structural = ['num_turns', 'total_tool_calls', 'subagent_messages', 
                 'total_errors', 'lead_agent_messages', 'num_subagents']
    
    # Quality variables
    quality = ['subagent_success_avg', 'subagent_similarity', 
              'subagents_completed_pct']
    
    # Outcome variables
    outcomes = ['accuracy', 'cost_usd', 'time_seconds']
    
    all_vars = outcomes + structural + quality
    
    # Select available columns
    available_vars = [v for v in all_vars if v in df.columns]
    
    # Compute correlation matrix
    corr_matrix = df[available_vars].corr()
    
    print("\nCorrelations with Accuracy:")
    if 'accuracy' in corr_matrix.index:
        acc_corr = corr_matrix['accuracy'].drop('accuracy').sort_values(key=abs, ascending=False)
        print(acc_corr.head(10).to_string())
    
    print("\nCorrelations with Cost:")
    if 'cost_usd' in corr_matrix.index:
        cost_corr = corr_matrix['cost_usd'].drop('cost_usd').sort_values(key=abs, ascending=False)
        print(cost_corr.head(10).to_string())
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=0.5)
    plt.title('Correlation Heatmap: Structural & Quality Metrics', fontsize=14, pad=20)
    plt.tight_layout()
    
    # Save figure
    Path(output_dir).mkdir(exist_ok=True)
    output_path = Path(output_dir) / 'correlation_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nHeatmap saved to: {output_path}")
    plt.close()
    
    results = {
        'correlation_matrix': corr_matrix,
        'output_path': str(output_path)
    }
    
    return results


def experiment_4_pca_analysis(df: pd.DataFrame, output_dir: str = 'experiments') -> dict:
    """
    Task 4: PCA - Operational Modes of the System
    Identify behavioral clusters with 2D PCA biplot.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 4: PCA - Operational Modes Analysis")
    print("="*80)
    
    # Input variables for PCA
    pca_vars = ['num_subagents', 'num_turns', 'lead_agent_messages', 
               'subagent_messages', 'total_tool_calls', 'total_errors', 
               'total_tokens', 'subagent_similarity', 'subagent_success_avg',
               'websearch_calls', 'webfetch_calls', 'cost_usd', 'time_seconds']
    
    # Prepare data
    df_clean = df.dropna(subset=pca_vars)
    
    # If too much missing data, impute with mean
    if len(df_clean) < len(df) * 0.5:
        print("Warning: Using mean imputation for PCA")
        df_clean = df.copy()
        for col in pca_vars:
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)
    
    X = df_clean[pca_vars]
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Explained variance
    print(f"\nExplained Variance:")
    print(f"  PC1: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"  PC2: {pca.explained_variance_ratio_[1]:.2%}")
    print(f"  Total: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Component loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=pca_vars
    )
    
    print(f"\nTop PC1 Loadings (Activity Scale):")
    print(loadings['PC1'].abs().sort_values(ascending=False).head(5).to_string())
    
    print(f"\nTop PC2 Loadings (Error/Coordination):")
    print(loadings['PC2'].abs().sort_values(ascending=False).head(5).to_string())
    
    # Create biplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Color by accuracy
    ax = axes[0]
    scatter1 = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                         c=df_clean['accuracy'], cmap='RdYlGn',
                         s=100, alpha=0.6, edgecolors='black')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
    ax.set_title('PCA Biplot - Colored by Accuracy', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax, label='Accuracy')
    
    # Plot 2: Color by run_type
    ax = axes[1]
    colors = df_clean['run_type'].map({'eval-multiagent': 'blue', 'eval-singleagent': 'orange'})
    for run_type, color in [('eval-multiagent', 'blue'), ('eval-singleagent', 'orange')]:
        mask = df_clean['run_type'] == run_type
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                  c=color, label=run_type, s=100, alpha=0.6, edgecolors='black')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=12)
    ax.set_title('PCA Biplot - Colored by Run Type', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    Path(output_dir).mkdir(exist_ok=True)
    output_path = Path(output_dir) / 'pca_biplot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nBiplot saved to: {output_path}")
    plt.close()
    
    results = {
        'pca': pca,
        'loadings': loadings,
        'components': X_pca,
        'explained_variance': pca.explained_variance_ratio_,
        'output_path': str(output_path)
    }
    
    return results


def run_all_experiments(csv_path: str, output_dir: str = 'experiments') -> dict:
    """Run all four experiments sequentially."""
    print("\n" + "="*80)
    print("RUNNING ALL EXPERIMENTS")
    print("="*80)
    
    # Load data
    df = load_dataset(csv_path)
    print(f"\nLoaded dataset: {len(df)} runs")
    print(f"Run types: {df['run_type'].value_counts().to_dict()}")
    
    # Run experiments
    results = {}
    
    results['experiment_1'] = experiment_1_logistic_regression(df)
    results['experiment_2'] = experiment_2_linear_regression(df)
    results['experiment_3'] = experiment_3_correlation_analysis(df, output_dir)
    results['experiment_4'] = experiment_4_pca_analysis(df, output_dir)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*80)
    
    return results


if __name__ == "__main__":
    # Run all experiments
    csv_path = "data/runww25.csv"
    output_dir = "experiments"
    
    results = run_all_experiments(csv_path, output_dir)
