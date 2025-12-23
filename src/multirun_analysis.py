## File largely written by GitHub Copilot running Claude Sonnet 4.5

import json
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from browsecomp_openai.samplers import ChatCompletionSampler

import re
import time


from dotenv import load_dotenv  # type: ignore
import os

load_dotenv()


_embedding_model = None
_llm_client = None
_llm_client_model = None

# Track evaluation failures globally
EVAL_STATS = {
    'total_attempted': 0,
    'successful': 0,
    'api_failures': 0,
    'parse_failures': 0,
    'empty_inputs': 0
}

SUBAGENT_GRADER_MODEL = "gpt-5-mini-2025-08-07"
SUBAGENT_GRADER_TEMPLATE = """
Judge whether the following [response] resolves the [subtask] and rate how well it was completed.

[subtask]: {subtask}

[response]: {response}

Your judgement must follow exactly this format:

solved: yes|no
quality_score: 0.0-10.0
analysis: {{brief justification referencing the subtask requirements only}}

Guidelines:
- Answer solved: yes only if no meaningful work remains; otherwise answer no.
- quality_score should reward completeness, clarity, autonomy, and initiative even when solved is yes.
- Use the full 0-10 range (decimals allowed). A 10 denotes flawless execution without caveats.
""".strip()


def get_embedding_model():
    """Lazy load the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model


def get_llm_client(model_name: str = SUBAGENT_GRADER_MODEL):
    global _llm_client, _llm_client_model
    if _llm_client is None or _llm_client_model != model_name:
        _llm_client = ChatCompletionSampler(model=model_name)
        _llm_client_model = model_name
    return _llm_client


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


def calculate_subagent_success_cos(prompt: str, response: str, threshold: float = 0.5) -> Tuple[float, int]:
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
    print("Warning: Using cosine similarity for subagent success metric. Consider using LLM-based grading instead.")
    # print(f"Prompt: {prompt}  \n  Response: {response}\n")
    model = get_embedding_model()
    
    # Get embeddings
    prompt_embedding = model.encode([prompt])
    response_embedding = model.encode([response])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(prompt_embedding, response_embedding)[0][0]
    
    # Binary completion: 1 if similarity >= threshold, 0 otherwise
    completed = 1 if similarity >= threshold else 0
    
    return float(similarity), completed


def calculate_subagent_success(
    prompt: str,
    response: str,
    threshold: float = 0.5,
    model: str = SUBAGENT_GRADER_MODEL,
    max_retries: int = 3
) -> Tuple[Optional[float], Optional[int]]:
    """
    Use an LLM grader to determine solved vs. quality (0-10).
    Returns (quality_score, solved_flag) or (None, None) if evaluation fails.

    Retries on API failures with exponential backoff.
    Tracks failures vs legitimate low scores via EVAL_STATS.
    """
    # print(prompt,'\n\n')
    _ = threshold  # retained for signature compatibility

    EVAL_STATS['total_attempted'] += 1

    if not prompt or not response:
        EVAL_STATS['empty_inputs'] += 1
        return None, None  # Return None instead of 0 to indicate failure

    grader = get_llm_client(model)
    grader_prompt = SUBAGENT_GRADER_TEMPLATE.format(subtask=prompt, response=response)
    prompt_messages = [grader._pack_message(content=grader_prompt, role="user")]

    # Retry logic with exponential backoff
    grading_response = None
    for attempt in range(max_retries):
        try:
            grading_response = grader(prompt_messages).response_text.strip()
            break  # Success, exit retry loop
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"API error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                # Final attempt failed
                EVAL_STATS['api_failures'] += 1
                print(f"API error after {max_retries} attempts: {e}")
                return None, None  # Return None to indicate failure

    if not grading_response:
        EVAL_STATS['api_failures'] += 1
        return None, None

    # Parse response - try multiple patterns for robustness
    solved = 0
    quality = 0.0

    # Pattern 1: Standard format
    solved_match = re.search(r"solved\s*:\s*(yes|no|1|0)", grading_response, flags=re.IGNORECASE)
    if solved_match:
        solved = 1 if solved_match.group(1).lower() in {"yes", "1"} else 0

    # Pattern 2: Try alternative patterns (handles non-English or variations)
    if not solved_match:
        # Try just finding yes/no/1/0 near "solved"
        if re.search(r"solved.*?(yes|1)", grading_response[:200], flags=re.IGNORECASE | re.DOTALL):
            solved = 1
        elif re.search(r"solved.*?(no|0)", grading_response[:200], flags=re.IGNORECASE | re.DOTALL):
            solved = 0

    quality_match = re.search(
        r"quality[_\s-]*score\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        grading_response,
        flags=re.IGNORECASE
    )
    if quality_match:
        try:
            quality = float(quality_match.group(1))
        except ValueError:
            pass

    # If we couldn't parse anything meaningful, it's a parse failure
    if quality == 0.0 and solved == 0 and not (solved_match or quality_match):
        EVAL_STATS['parse_failures'] += 1
        print(f"Parse failure - response: {grading_response[:200]}")
        return None, None

    EVAL_STATS['successful'] += 1
    quality = max(0.0, min(10.0, quality))
    return quality, solved


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
        'correctness': metrics['overview'].get('correctness'),
        'confidence': metrics['overview'].get('confidence'),
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

        num_subagents = len(run_data['subagent_prompts'])
        for idx, (prompt, response) in enumerate(zip(run_data['subagent_prompts'], run_data['subagent_responses']), 1):
            # Progress indicator
            print(f"[Progress] Evaluating subagent {EVAL_STATS['total_attempted']+1} (run subagent {idx}/{num_subagents})...", flush=True)
            try:
                score, completed = calculate_subagent_success(prompt, response)
                # Skip None values (evaluation failures) - don't include in averages
                if score is not None and completed is not None:
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
                'correctness': run_data['correctness'],
                'confidence': run_data['confidence'],
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


def print_evaluation_stats():
    """Print statistics about subagent evaluation failures."""
    print("\n" + "=" * 80)
    print("SUBAGENT EVALUATION STATISTICS")
    print("=" * 80)
    print(f"Total subagents attempted:     {EVAL_STATS['total_attempted']}")
    print(f"Successfully evaluated:        {EVAL_STATS['successful']}")
    print(f"Failed - API errors:           {EVAL_STATS['api_failures']}")
    print(f"Failed - Parse errors:         {EVAL_STATS['parse_failures']}")
    print(f"Skipped - Empty inputs:        {EVAL_STATS['empty_inputs']}")

    failed_total = EVAL_STATS['api_failures'] + EVAL_STATS['parse_failures'] + EVAL_STATS['empty_inputs']
    if EVAL_STATS['total_attempted'] > 0:
        success_rate = 100 * EVAL_STATS['successful'] / EVAL_STATS['total_attempted']
        failure_rate = 100 * failed_total / EVAL_STATS['total_attempted']
        print(f"\nSuccess rate:                  {success_rate:.1f}%")
        print(f"Failure rate:                  {failure_rate:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    # Example usage
    path = "ww100"
    results_path = f"data/{path}.json"
    dataset = load_and_create_dataset(results_path)

    # Print evaluation statistics (failures vs legitimate scores)
    print_evaluation_stats()

    # Print summary
    print("\nDataset Summary:")
    print("=" * 80)
    print(f"Total runs: {len(dataset)}")
    print(f"\nBy run type:")
    for run_type in dataset['run_type'].unique():
        subset = dataset[dataset['run_type'] == run_type]
        print(f"\n{run_type}: {len(subset)} runs")
        print(f"  Avg accuracy: {subset['accuracy'].mean():.2f}")
        print(f"  Avg correctness: {subset['correctness'].mean():.2f}")
        print(f"  Avg confidence: {subset['confidence'].mean():.2f}%")
        print(f"  Avg time: {subset['time_seconds'].mean():.2f}s")
        print(f"  Avg tokens: {subset['total_tokens'].mean():.0f}")
        print(f"  Avg cost: ${subset['cost_usd'].mean():.4f}")
        print(f"  Avg subagents: {subset['num_subagents'].mean():.1f}")

    # Save dataset
    save_dataset(dataset, f"data/{path}.csv")
    print(f"\nDataset saved to data/{path}.csv")

    # Also print basic stats
    print("\n" + "=" * 80)
    print("Dataset Info:")
    print(dataset.info())
    print("\nFirst few rows:")
    print(dataset.head())