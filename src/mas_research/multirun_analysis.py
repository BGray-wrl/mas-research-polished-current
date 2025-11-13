## File largely written by GitHub Copilot running Claude Sonnet 4.5

import json
from typing import Dict, List, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load embedding model once at module level
_embedding_model = None

def get_embedding_model():
    """Lazy load the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model


def calculate_subagent_similarity(initializing_prompts: List[str]) -> float:
    """
    Calculate similarity metric for a list of subagent initializing prompts.
    Uses average pairwise cosine similarity between prompt embeddings.
    
    Args:
        initializing_prompts: List of prompt strings from all subagents
        
    Returns:
        float: Average pairwise cosine similarity (0.0 to 1.0)
    """
    if len(initializing_prompts) < 2:
        return 1.0  # Single prompt is perfectly similar to itself
    
    model = get_embedding_model()
    embeddings = model.encode(initializing_prompts)
    
    # Calculate pairwise cosine similarities
    similarities = cosine_similarity(embeddings)
    
    # Get upper triangle (excluding diagonal) to avoid counting each pair twice
    n = len(similarities)
    upper_triangle = similarities[np.triu_indices(n, k=1)]
    
    # Return average similarity
    return float(np.mean(upper_triangle))


def calculate_subagent_success(prompt: str, response: str, threshold: float = 0.5) -> Tuple[float, int]:
    """
    Calculate success metric for a single subagent based on its prompt and response.
    Measures cosine similarity between prompt and response embeddings.
    Higher similarity suggests the response is relevant to the prompt.
    
    Args:
        prompt: The initializing prompt for the subagent
        response: The response/return text from the subagent
        threshold: Similarity threshold for binary completion (default 0.5)
        
    Returns:
        Tuple[float, int]: (similarity score 0.0-1.0, binary completion 0 or 1)
    """
    model = get_embedding_model()
    
    # Get embeddings
    prompt_embedding = model.encode([prompt])
    response_embedding = model.encode([response])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(prompt_embedding, response_embedding)[0][0]
    
    # Binary completion: 1 if similarity >= threshold, 0 otherwise
    completed = 1 if similarity >= threshold else 0
    
    return float(similarity), completed



def normalize_grade(grade: Any) -> int:
    """
    Convert grade to binary 0/1 format.
    
    Args:
        grade: Grade value (could be 0/1, 'yes'/'no', True/False, etc.)
        
    Returns:
        int: 0 or 1
    """
    if isinstance(grade, bool):
        return 1 if grade else 0
    if isinstance(grade, int):
        return grade
    if isinstance(grade, str):
        grade_lower = grade.lower().strip()
        if grade_lower in ['yes', 'correct', '1', 'true', 'pass']:
            return 1
        elif grade_lower in ['no', 'incorrect', '0', 'false', 'fail']:
            return 0
    return 0


def extract_subagent_clustering(metrics: Dict[str, Any]) -> List[float]:
    """
    Extract clustering metric showing when subagents were initialized.
    Returns list of percentages (0.0 to 1.0) indicating initialization points.
    
    Args:
        metrics: Single run metrics from analyze_agent_metrics
        
    Returns:
        List[float]: List of initialization percentages
    """
    total_tool_calls = metrics['tools']['total_tool_calls']
    if total_tool_calls == 0:
        return []
    
    clustering = []
    for agent_id, agent_data in metrics['agents']['by_agent'].items():
        if agent_id == 'lead_agent':
            continue
        init_at = agent_data.get('initialized_at_tool_call')
        if init_at is not None:
            clustering.append(init_at / total_tool_calls)
    
    return sorted(clustering)


def extract_run_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key metrics from a single run.
    
    Args:
        metrics: Output from analyze_agent_metrics
        
    Returns:
        Dictionary with extracted metrics
    """
    # Basic accuracy
    grade = normalize_grade(metrics['overview']['grade'])
    
    # Time and cost metrics
    time_seconds = metrics['costs']['duration_seconds']
    total_tokens = metrics['costs']['token_usage']['total_tokens']
    cost_usd = metrics['costs']['total_cost_usd']
    num_turns = metrics['costs']['num_turns']
    
    # Subagent counts
    num_subagents = metrics['agents']['unique_subagent_instances']
    total_messages = metrics['overview']['total_messages']
    
    # Message breakdown by agent level
    lead_agent_messages = metrics['message_breakdown']['by_agent_level']['lead_agent']
    subagent_messages = metrics['message_breakdown']['by_agent_level']['subagent']
    
    # Tool usage metrics
    total_errors = metrics['tools']['errors']
    total_tool_calls = metrics['tools']['total_tool_calls']
    total_subagent_calls = metrics['agents']['total_subagent_calls']
    
    # Individual tool counts
    tool_counts = {
        'websearch_calls': metrics['tools']['by_tool'].get('WebSearch', 0),
        'webfetch_calls': metrics['tools']['by_tool'].get('WebFetch', 0),
        'read_calls': metrics['tools']['by_tool'].get('Read', 0),
        'task_calls': metrics['tools']['by_tool'].get('Task', 0),
        'bash_calls': metrics['tools']['by_tool'].get('Bash', 0),
    }
    
    # Extract subagent prompts and responses for similarity/success
    subagent_prompts = []
    subagent_responses = []
    for agent_id, agent_data in metrics['agents']['by_agent'].items():
        if agent_id == 'lead_agent':
            continue
        init_prompt_info = agent_data.get('initializing_prompt')
        if init_prompt_info:
            subagent_prompts.append(init_prompt_info['prompt'])
        return_text = agent_data.get('return_text')
        if return_text:
            subagent_responses.append(return_text)
    
    # Clustering metric
    clustering = extract_subagent_clustering(metrics)
    
    return {
        'accuracy': grade,
        'time_seconds': time_seconds,
        'total_tokens': total_tokens,
        'cost_usd': cost_usd,
        'num_turns': num_turns,
        'num_subagents': num_subagents,
        'total_messages': total_messages,
        'lead_agent_messages': lead_agent_messages,
        'subagent_messages': subagent_messages,
        'total_errors': total_errors,
        'total_tool_calls': total_tool_calls,
        'total_subagent_calls': total_subagent_calls,
        **tool_counts,
        'subagent_prompts': subagent_prompts,
        'subagent_responses': subagent_responses,
        'subagent_clustering': clustering,
        'question': metrics['overview']['question'],
        'expected_answer': metrics['overview']['expected_answer'],
        'received_answer': metrics['overview']['received_answer'],
    }


def calculate_subagent_metrics(run_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate subagent similarity and success metrics for a run.
    
    Args:
        run_data: Output from extract_run_metrics
        
    Returns:
        Dictionary with subagent_similarity, subagent_success_avg, completion metrics
    """
    result = {}
    
    # Similarity metric
    if len(run_data['subagent_prompts']) > 0:
        try:
            result['subagent_similarity'] = calculate_subagent_similarity(
                run_data['subagent_prompts']
            )
        except NotImplementedError:
            result['subagent_similarity'] = None
    else:
        result['subagent_similarity'] = None
    
    # Success metrics (average across all subagents)
    if len(run_data['subagent_prompts']) > 0:
        success_scores = []
        completions = []
        
        for prompt, response in zip(run_data['subagent_prompts'], run_data['subagent_responses']):
            try:
                score, completed = calculate_subagent_success(prompt, response)
                success_scores.append(score)
                completions.append(completed)
            except NotImplementedError:
                pass
        
        result['subagent_success_avg'] = sum(success_scores) / len(success_scores) if success_scores else None
        result['subagents_completed'] = sum(completions) if completions else 0
        result['subagents_completed_pct'] = sum(completions) / len(completions) if completions else None
    else:
        result['subagent_success_avg'] = None
        result['subagents_completed'] = 0
        result['subagents_completed_pct'] = None
    
    return result


def create_dataset(results: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    Create dataset from results structure (output of assemble_run_metrics.py).
    
    Args:
        results: Dictionary mapping run_type to list of metrics
        
    Returns:
        DataFrame with all runs and their metrics
    """
    all_records = []
    
    for run_type, runs in results.items():
        for i, run_metrics in enumerate(runs):
            # Extract basic metrics
            run_data = extract_run_metrics(run_metrics)
            
            # Calculate subagent metrics
            subagent_metrics = calculate_subagent_metrics(run_data)
            
            # Calculate clustering statistics
            clustering = run_data['subagent_clustering']
            clustering_mean = sum(clustering) / len(clustering) if clustering else None
            clustering_std = None
            if clustering and len(clustering) > 1:
                mean = clustering_mean
                variance = sum((x - mean) ** 2 for x in clustering) / len(clustering)
                clustering_std = variance ** 0.5
            
            # Combine into single record (excluding long strings)
            record = {
                'run_type': run_type,
                'run_index': i,
                'accuracy': run_data['accuracy'],
                'time_seconds': run_data['time_seconds'],
                'total_tokens': run_data['total_tokens'],
                'cost_usd': run_data['cost_usd'],
                'num_turns': run_data['num_turns'],
                'num_subagents': run_data['num_subagents'],
                'total_messages': run_data['total_messages'],
                'lead_agent_messages': run_data['lead_agent_messages'],
                'subagent_messages': run_data['subagent_messages'],
                'total_errors': run_data['total_errors'],
                'total_tool_calls': run_data['total_tool_calls'],
                'total_subagent_calls': run_data['total_subagent_calls'],
                'websearch_calls': run_data['websearch_calls'],
                'webfetch_calls': run_data['webfetch_calls'],
                'read_calls': run_data['read_calls'],
                'task_calls': run_data['task_calls'],
                'bash_calls': run_data['bash_calls'],
                'subagent_similarity': subagent_metrics['subagent_similarity'],
                'subagent_success_avg': subagent_metrics['subagent_success_avg'],
                'subagents_completed': subagent_metrics['subagents_completed'],
                'subagents_completed_pct': subagent_metrics['subagents_completed_pct'],
                'clustering_mean': clustering_mean,
                'clustering_std': clustering_std,
                'num_clusters': len(clustering),
            }
            
            all_records.append(record)
    
    return pd.DataFrame(all_records)


def load_and_create_dataset(results_path: str) -> pd.DataFrame:
    """
    Load results from JSON file and create dataset.
    
    Args:
        results_path: Path to JSON file with results
        
    Returns:
        DataFrame with dataset
    """
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return create_dataset(results)


def save_dataset(dataset: pd.DataFrame, output_path: str):
    """
    Save dataset to CSV file.
    
    Args:
        dataset: Dataset DataFrame
        output_path: Path to save CSV file
    """
    dataset.to_csv(output_path, index=False)


if __name__ == "__main__":
    # Example usage
    results_path = "data/cur.json"
    dataset = load_and_create_dataset(results_path)
    
    # Print summary
    print("\nDataset Summary:")
    print("=" * 80)
    print(f"Total runs: {len(dataset)}")
    print(f"\nBy run type:")
    for run_type in dataset['run_type'].unique():
        subset = dataset[dataset['run_type'] == run_type]
        print(f"\n{run_type}: {len(subset)} runs")
        print(f"  Avg accuracy: {subset['accuracy'].mean():.2f}")
        print(f"  Avg time: {subset['time_seconds'].mean():.2f}s")
        print(f"  Avg tokens: {subset['total_tokens'].mean():.0f}")
        print(f"  Avg cost: ${subset['cost_usd'].mean():.4f}")
        print(f"  Avg subagents: {subset['num_subagents'].mean():.1f}")
    
    # Save dataset
    save_dataset(dataset, "data/dataset.csv")
    print("\nDataset saved to data/dataset.csv")
    
    # Also print basic stats
    print("\n" + "=" * 80)
    print("Dataset Info:")
    print(dataset.info())
    print("\nFirst few rows:")
    print(dataset.head())