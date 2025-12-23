## Multi-Agent Analysis Script
## Focuses on internal properties and dynamics of multi-agent runs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


def load_multiagent_data(csv_path: str) -> pd.DataFrame:
    """Load dataset and filter for multi-agent runs only."""
    df = pd.read_csv(csv_path)
    df_multi = df[df['run_type'] == 'ww-eval-multiagent'].copy()
    print(f"Loaded {len(df_multi)} multi-agent runs from {len(df)} total runs")
    return df_multi


def experiment_1_coordination_success_analysis(df: pd.DataFrame) -> dict:
    """
    Experiment 1: Multi-Agent Coordination and Success
    Analyze how subagent coordination metrics predict task success.
    Focus on num_subagents, subagent_similarity, subagent_success_avg, and completion rates.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 1: Multi-Agent Coordination and Success Analysis")
    print("="*80)
    
    # Coordination features
    coordination_features = [
        'num_subagents', 
        'subagent_similarity', 
        'subagent_success_avg',
        'subagents_completed_pct',
    ]
    
    # Activity features that may affect coordination
    activity_features = [
        'num_turns',
        'total_tool_calls',
        'total_errors',
        'websearch_calls',
        'webfetch_calls'
    ]
    
    all_features = coordination_features + activity_features
    
    # Clean data
    df_clean = df.dropna(subset=['accuracy'])
    
    # Fill NaN in features (some runs may have missing coordination metrics)
    for col in all_features:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)
    
    # Check available features
    available_features = [f for f in all_features if f in df_clean.columns]
    X = df_clean[available_features]
    y = df_clean['accuracy']
    
    print(f"\nDataset: {len(df_clean)} runs")
    print(f"Accuracy distribution: {y.value_counts().to_dict()}")
    print(f"\nCoordination Metrics Summary:")
    if 'num_subagents' in df_clean.columns:
        print(f"  Subagents per run: mean={df_clean['num_subagents'].mean():.2f}, std={df_clean['num_subagents'].std():.2f}")
    if 'subagent_similarity' in df_clean.columns:
        print(f"  Subagent similarity: mean={df_clean['subagent_similarity'].mean():.3f}, std={df_clean['subagent_similarity'].std():.3f}")
    if 'subagent_success_avg' in df_clean.columns:
        print(f"  Subagent success avg: mean={df_clean['subagent_success_avg'].mean():.2f}, std={df_clean['subagent_success_avg'].std():.2f}")
    
    # Check if we have variation in accuracy
    if len(y.unique()) < 2:
        print(f"\nWarning: Only one class present (all {y.iloc[0]})")
        print("Showing feature correlations with other metrics instead:")
        
        # Correlations with cost and time
        for target in ['cost_usd', 'time_seconds']:
            if target in df_clean.columns:
                print(f"\nCorrelations with {target}:")
                correlations = df_clean[available_features + [target]].corr()[target].drop(target).sort_values(key=abs, ascending=False)
                print(correlations.head(5).to_string())
        
        return {
            'model': None,
            'accuracy': None,
            'features': available_features,
            'note': 'Insufficient class variation'
        }
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit logistic regression
    model = LogisticRegression(random_state=42, max_iter=1000, penalty='l2', C=0.1)
    model.fit(X_scaled, y)
    
    # Predictions
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]
    
    accuracy = (y_pred == y).mean()
    
    # Feature importance
    coefficients = pd.DataFrame({
        'feature': available_features,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print(f"\nModel Accuracy: {accuracy:.2%}")
    print(f"\nTop Coordination Features for Success:")
    print(coefficients.head(10).to_string(index=False))
    
    return {
        'model': model,
        'accuracy': accuracy,
        'coefficients': coefficients,
        'predictions': y_pred,
        'probabilities': y_prob,
        'features': available_features
    }


def experiment_2_resource_efficiency_analysis(df: pd.DataFrame) -> dict:
    """
    Experiment 2: Resource Efficiency in Multi-Agent Systems
    Predict cost and time based on coordination complexity and activity.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 2: Multi-Agent Resource Efficiency Analysis")
    print("="*80)
    
    # Features related to multi-agent coordination overhead
    predictors = [
        'num_subagents',
        'subagent_messages',
        'num_turns',
        'total_tool_calls',
        'websearch_calls',
        'webfetch_calls',
        'total_errors',
        'subagent_similarity'  # Higher similarity might indicate redundant work
    ]
    
    results = {}
    
    for target in ['cost_usd', 'time_seconds']:
        print(f"\n--- Predicting {target} ---")
        
        # Clean data
        df_clean = df.dropna(subset=[target])
        
        # Check available predictors
        available_predictors = [p for p in predictors if p in df_clean.columns]
        
        # Fill NaN in predictors
        for col in available_predictors:
            df_clean[col] = df_clean[col].fillna(0)
        
        X = df_clean[available_predictors]
        y = df_clean[target]
        
        # Standardize
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
            'feature': available_predictors,
            'coefficient': model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
        
        print(f"R² Score: {r2:.3f}")
        print(f"RMSE: {rmse:.3f}")
        print(f"Mean {target}: {y.mean():.3f}")
        print(f"\nTop Predictors:")
        print(coefficients.head(10).to_string(index=False))
        
        results[target] = {
            'model': model,
            'r2': r2,
            'rmse': rmse,
            'coefficients': coefficients,
            'predictions': y_pred,
            'actual': y.values
        }
    
    return results


def experiment_3_coordination_patterns_correlation(df: pd.DataFrame, output_dir: str = 'experiments/visuals') -> dict:
    """
    Experiment 3: Multi-Agent Coordination Patterns
    Correlation analysis focused on coordination metrics.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 3: Multi-Agent Coordination Pattern Analysis")
    print("="*80)
    
    # Multi-agent specific metrics
    coordination_vars = [
        'num_subagents',
        'subagent_similarity',
        'subagent_success_avg',
        'subagents_completed_pct',
        'subagent_messages',
        'lead_agent_messages'
    ]
    
    # Activity metrics
    activity_vars = [
        'num_turns',
        'total_tool_calls',
        'websearch_calls',
        'webfetch_calls',
        'total_errors'
    ]
    
    # Outcomes
    outcome_vars = [
        'accuracy',
        'cost_usd',
        'time_seconds',
        'total_tokens'
    ]
    
    all_vars = outcome_vars + coordination_vars + activity_vars
    
    # Filter for available columns
    available_vars = [v for v in all_vars if v in df.columns]
    
    # Compute correlation matrix
    corr_matrix = df[available_vars].corr()
    
    # Analyze key relationships
    print("\nKey Coordination Correlations:")
    print("\n1. Correlations with Accuracy:")
    if 'accuracy' in corr_matrix.index:
        acc_corr = corr_matrix['accuracy'].drop('accuracy').sort_values(key=abs, ascending=False)
        print(acc_corr.head(8).to_string())
    
    print("\n2. Correlations with Cost:")
    if 'cost_usd' in corr_matrix.index:
        cost_corr = corr_matrix['cost_usd'].drop('cost_usd').sort_values(key=abs, ascending=False)
        print(cost_corr.head(8).to_string())
    
    print("\n3. Subagent Coordination Relationships:")
    coord_subset = [v for v in coordination_vars if v in available_vars]
    if len(coord_subset) > 1:
        coord_corr = df[coord_subset].corr()
        print("\nSubagent metrics intercorrelations:")
        print(coord_corr.to_string())
    
    # Create heatmap
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Full correlation heatmap
    ax = axes[0]
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Correlation'})
    ax.set_title('Multi-Agent System: Full Correlation Matrix', fontsize=14, pad=20)
    
    # Coordination-focused heatmap
    ax = axes[1]
    coord_outcome_vars = [v for v in coordination_vars + outcome_vars if v in available_vars]
    if len(coord_outcome_vars) > 1:
        coord_matrix = df[coord_outcome_vars].corr()
        sns.heatmap(coord_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=0.5, ax=ax,
                   cbar_kws={'label': 'Correlation'})
        ax.set_title('Multi-Agent Coordination Metrics Focus', fontsize=14, pad=20)
    
    plt.tight_layout()
    
    # Save figure
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / 'multiagent_correlation_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nHeatmap saved to: {output_path}")
    plt.close()
    
    return {
        'correlation_matrix': corr_matrix,
        'output_path': str(output_path)
    }


def experiment_4_operational_clustering(df: pd.DataFrame, output_dir: str = 'experiments/visuals') -> dict:
    """
    Experiment 4: Multi-Agent Operational Modes via PCA & Clustering
    Identify different operational patterns in multi-agent coordination.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 4: Multi-Agent Operational Modes (PCA & Clustering)")
    print("="*80)
    
    # Multi-agent operational features
    pca_vars = [
        'num_subagents',
        'subagent_messages',
        'lead_agent_messages',
        'num_turns',
        'total_tool_calls',
        'websearch_calls',
        'webfetch_calls',
        'total_errors',
        'subagent_similarity',
        'subagent_success_avg',
        'subagents_completed_pct',
        'cost_usd',
        'time_seconds',
        'total_tokens', 
        'accuracy'
    ]
    
    # Clean data
    available_vars = [v for v in pca_vars if v in df.columns]
    df_clean = df.copy()
    
    # Fill NaN with mean
    for col in available_vars:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    
    X = df_clean[available_vars]
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Explained variance
    print(f"\nPCA Explained Variance:")
    print(f"  PC1: {pca.explained_variance_ratio_[0]:.2%}")
    print(f"  PC2: {pca.explained_variance_ratio_[1]:.2%}")
    print(f"  Total: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Component loadings
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=available_vars
    )
    
    print(f"\nTop PC1 Loadings:")
    print(loadings['PC1'].abs().sort_values(ascending=False).head(6).to_string())
    
    print(f"\nTop PC2 Loadings:")
    print(loadings['PC2'].abs().sort_values(ascending=False).head(6).to_string())
    
    # K-means clustering to identify operational modes
    n_clusters = 3  # Try 3 operational modes
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca)
    
    df_clean['cluster'] = clusters
    
    print(f"\nOperational Mode Distribution:")
    print(df_clean['cluster'].value_counts().sort_index().to_string())
    
    # Analyze cluster characteristics
    print(f"\nCluster Characteristics:")
    cluster_means = df_clean.groupby('cluster')[available_vars].mean()
    print(cluster_means.to_string())
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot 1: PCA colored by accuracy
    ax = axes[0, 0]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], 
                        c=df_clean['accuracy'], cmap='RdYlGn',
                        s=120, alpha=0.7, edgecolors='black', linewidth=1.5)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
    ax.set_title('Multi-Agent Runs by Accuracy', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Accuracy')
    
    # Plot 2: PCA colored by operational cluster
    ax = axes[0, 1]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for i in range(n_clusters):
        mask = clusters == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                  c=colors[i], label=f'Mode {i+1}', 
                  s=120, alpha=0.7, edgecolors='black', linewidth=1.5)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
    ax.set_title('Multi-Agent Operational Modes', fontsize=13, fontweight='bold')
    ax.legend(title='Operational Mode')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: PCA colored by number of subagents
    ax = axes[1, 0]
    if 'num_subagents' in df_clean.columns:
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                           c=df_clean['num_subagents'], cmap='viridis',
                           s=120, alpha=0.7, edgecolors='black', linewidth=1.5)
        plt.colorbar(scatter, ax=ax, label='Number of Subagents')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
    ax.set_title('Multi-Agent Runs by Subagent Count', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: PCA colored by cost
    ax = axes[1, 1]
    if 'cost_usd' in df_clean.columns:
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                           c=df_clean['cost_usd'], cmap='plasma',
                           s=120, alpha=0.7, edgecolors='black', linewidth=1.5)
        plt.colorbar(scatter, ax=ax, label='Cost (USD)')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
    ax.set_title('Multi-Agent Runs by Cost', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / 'multiagent_pca_clustering.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPCA clustering plot saved to: {output_path}")
    plt.close()
    
    return {
        'pca': pca,
        'loadings': loadings,
        'components': X_pca,
        'clusters': clusters,
        'cluster_means': cluster_means,
        'explained_variance': pca.explained_variance_ratio_,
        'output_path': str(output_path)
    }


def experiment_5_subagent_coordination_dynamics(df: pd.DataFrame, output_dir: str = 'experiments/visuals') -> dict:
    """
    Experiment 5: Subagent Coordination Dynamics
    Deep dive into how subagent coordination affects outcomes.
    """
    print("\n" + "="*80)
    print("EXPERIMENT 5: Subagent Coordination Dynamics")
    print("="*80)
    
    # Focus on coordination-specific relationships
    coord_metrics = ['num_subagents', 'subagent_similarity', 'subagent_success_avg', 
                     'subagents_completed_pct']
    
    available_metrics = [m for m in coord_metrics if m in df.columns and df[m].notna().sum() > 0]
    
    if len(available_metrics) < 2:
        print("Insufficient coordination metrics available")
        return {'note': 'Insufficient data'}
    
    # Create scatter plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    plot_idx = 0
    
    # Plot coordination metrics vs accuracy
    if 'accuracy' in df.columns:
        # Define x-jitter strengths based on metric ranges
        # num_subagents: 1-3 (range 2), subagent_similarity: 0.4-1 (range 0.6), subagent_success_avg: 0-10 (range 10)
        x_jitter_map = {
            'num_subagents': 0.08,           # ~4% of range 2
            'subagent_similarity': 0.02,      # ~3% of range 0.6
            'subagent_success_avg': 0.3,      # ~3% of range 10
            'subagents_completed_pct': 0.03   # ~3% of range 1
        }
        
        for metric in available_metrics[:3]:
            ax = axes[plot_idx]
            
            # Separate by accuracy
            success = df[df['accuracy'] == 1]
            failure = df[df['accuracy'] == 0]
            
            # Add jitter to both x and y coordinates to prevent overlap
            y_jitter_strength = 0.08
            x_jitter_strength = x_jitter_map.get(metric, 0.05)
            
            success_y_jitter = 1 + np.random.uniform(-y_jitter_strength, y_jitter_strength, len(success))
            failure_y_jitter = 0 + np.random.uniform(-y_jitter_strength, y_jitter_strength, len(failure))
            
            success_x_jitter = success[metric] + np.random.uniform(-x_jitter_strength, x_jitter_strength, len(success))
            failure_x_jitter = failure[metric] + np.random.uniform(-x_jitter_strength, x_jitter_strength, len(failure))
            
            ax.scatter(success_x_jitter, success_y_jitter, 
                      c='green', s=100, alpha=0.6, label='Success', marker='o')
            ax.scatter(failure_x_jitter, failure_y_jitter, 
                      c='red', s=100, alpha=0.6, label='Failure', marker='x')
            
            ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=10)
            ax.set_ylabel('Accuracy', fontsize=10)
            ax.set_yticks([0, 1])
            ax.set_ylim(-0.15, 1.15)
            ax.set_title(f'{metric.replace("_", " ").title()} vs Success', fontsize=11)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
    
    # Plot coordination metrics vs cost
    if 'cost_usd' in df.columns and plot_idx < 6:
        for metric in available_metrics[:min(3, 6-plot_idx)]:
            ax = axes[plot_idx]
            
            valid_data = df[[metric, 'cost_usd']].dropna()
            
            # Add jitter to x-coordinates to prevent overlap
            x_jitter_strength = x_jitter_map.get(metric, 0.05)
            x_jittered = valid_data[metric] + np.random.uniform(-x_jitter_strength, x_jitter_strength, len(valid_data))
            
            scatter = ax.scatter(x_jittered, valid_data['cost_usd'],
                               c=valid_data['cost_usd'], cmap='viridis',
                               s=100, alpha=0.6, edgecolors='black')
            
            ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=10)
            ax.set_ylabel('Cost (USD)', fontsize=10)
            ax.set_title(f'{metric.replace("_", " ").title()} vs Cost', fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Add correlation coefficient
            if len(valid_data) > 1:
                corr = valid_data[metric].corr(valid_data['cost_usd'])
                ax.text(0.05, 0.95, f'r = {corr:.3f}', 
                       transform=ax.transAxes, fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, 6):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / 'multiagent_coordination_dynamics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nCoordination dynamics plot saved to: {output_path}")
    plt.close()
    
    # Compute statistics
    print("\nCoordination Metric Statistics by Accuracy:")
    if 'accuracy' in df.columns:
        stats = df.groupby('accuracy')[available_metrics].agg(['mean', 'std', 'count'])
        print(stats.to_string())
    
    return {
        'output_path': str(output_path),
        'available_metrics': available_metrics
    }


def run_all_multiagent_experiments(csv_path: str, output_dir: str = 'experiments/visuals') -> dict:
    """Run all multi-agent focused experiments."""
    print("\n" + "="*80)
    print("RUNNING MULTI-AGENT ANALYSIS EXPERIMENTS")
    print("="*80)
    
    # Load multi-agent data
    df = load_multiagent_data(csv_path)
    
    if len(df) == 0:
        print("ERROR: No multi-agent runs found!")
        return {}
    
    print(f"\nAnalyzing {len(df)} multi-agent runs")
    print(f"Accuracy rate: {df['accuracy'].mean():.1%}")
    if 'num_subagents' in df.columns:
        print(f"Subagents per run: mean={df['num_subagents'].mean():.1f}, range=[{df['num_subagents'].min():.0f}, {df['num_subagents'].max():.0f}]")
    
    # Run experiments
    results = {}
    
    # results['experiment_1'] = experiment_1_coordination_success_analysis(df)
    # results['experiment_2'] = experiment_2_resource_efficiency_analysis(df)
    # results['experiment_3'] = experiment_3_coordination_patterns_correlation(df, output_dir)
    results['experiment_4'] = experiment_4_operational_clustering(df, output_dir)
    # results['experiment_5'] = experiment_5_subagent_coordination_dynamics(df, output_dir)
    
    print("\n" + "="*80)
    print("ALL MULTI-AGENT EXPERIMENTS COMPLETE")
    print(f"Visualizations saved to: {output_dir}/")
    print("="*80)
    
    return results


if __name__ == "__main__":
    # Run all multi-agent experiments
    csv_path = "data/ww100.csv" ## CHANGE ME to your desired dataset path
    output_dir = "experiments/visuals"
    
    results = run_all_multiagent_experiments(csv_path, output_dir)
