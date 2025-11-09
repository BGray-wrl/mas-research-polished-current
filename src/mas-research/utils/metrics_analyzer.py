from typing import Dict, List, Any
from collections import defaultdict, Counter


def analyze_agent_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze metrics from a multi-agent conversation run.
    
    Args:
        data: The loaded JSON data containing question, answer, and messages
        
    Returns:
        Dictionary with comprehensive metrics broken down by agent level
    """
    messages = data.get('messages', [])
    
    # Initialize metrics
    metrics = {
        'overview': {
            'question': data.get('question', 'N/A'),
            'expected_answer': data.get('expected_answer', 'N/A'),
            'received_answer': data.get('recieved_answer', 'N/A'),
            'total_messages': len(messages),
        },
        'agents': {
            'total_subagent_calls': 0,
            'subagent_types': [],
            'models_used': Counter(),
            'by_agent': {},  # NEW: per-agent metrics
        },
        'tools': {
            'total_tool_calls': 0,
            'by_tool': Counter(),
            'by_agent_level': {
                'lead_agent': Counter(),
                'subagents': Counter(),
            },
            'errors': 0,
            'error_details': [],
        },
        'costs': {
            'total_cost_usd': 0.0,
            'num_turns': 0,
            'duration_ms': 0,
            'token_usage': {
                'input_tokens': 0,
                'output_tokens': 0,
                'cache_creation_input_tokens': 0,
                'cache_read_input_tokens': 0,
                'total_tokens': 0,
            },
            'cache_creation': {},
            'web_search_requests': 0,
        },
        'message_breakdown': {
            'by_type': Counter(),
            'by_agent_level': {
                'lead_agent': 0,
                'subagent': 0,
            },
        },
    }
    
    # Track unique subagent IDs to count distinct subagent instances
    subagent_tool_use_ids = set()
    
    # Process each message
    for msg in messages:
        msg_type = msg.get('type', 'Unknown')
        msg_data = msg.get('data', {})
        
        metrics['message_breakdown']['by_type'][msg_type] += 1
        
        # Check if this is a lead agent or subagent message
        parent_tool_use_id = msg_data.get('parent_tool_use_id')
        is_subagent = parent_tool_use_id is not None
        
        # NEW: Determine agent identifier
        agent_id = parent_tool_use_id if is_subagent else 'lead_agent'
        
        # NEW: Initialize agent metrics if first time seeing this agent
        if agent_id not in metrics['agents']['by_agent']:
            metrics['agents']['by_agent'][agent_id] = {
                'messages': 0,
                'tool_calls': Counter(),
                'errors': 0,
                'model': None,
            }
        
        # NEW: Increment message count for this agent
        metrics['agents']['by_agent'][agent_id]['messages'] += 1
        
        if is_subagent:
            metrics['message_breakdown']['by_agent_level']['subagent'] += 1
        else:
            metrics['message_breakdown']['by_agent_level']['lead_agent'] += 1
        
        # Track model usage
        if 'model' in msg_data:
            model = msg_data['model']
            metrics['agents']['models_used'][model] += 1
            # NEW: Track model for this specific agent
            metrics['agents']['by_agent'][agent_id]['model'] = model
        
        # Process content blocks
        content = msg_data.get('content', [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block_type = block.get('type', '')
                    
                    # Count tool uses
                    if block_type == 'ToolUseBlock':
                        tool_name = block.get('name', 'Unknown')
                        metrics['tools']['total_tool_calls'] += 1
                        metrics['tools']['by_tool'][tool_name] += 1
                        
                        # NEW: Track tool use for this specific agent
                        metrics['agents']['by_agent'][agent_id]['tool_calls'][tool_name] += 1
                        
                        # Track if this is a subagent call
                        if tool_name == 'Task':
                            metrics['agents']['total_subagent_calls'] += 1
                            subagent_type = block.get('input', {}).get('subagent_type', 'Unknown')
                            if subagent_type not in metrics['agents']['subagent_types']:
                                metrics['agents']['subagent_types'].append(subagent_type)
                            subagent_tool_use_ids.add(block.get('id'))
                        
                        # Categorize by agent level
                        if is_subagent:
                            metrics['tools']['by_agent_level']['subagents'][tool_name] += 1
                        else:
                            metrics['tools']['by_agent_level']['lead_agent'][tool_name] += 1
                    
                    # Count errors
                    elif block_type == 'ToolResultBlock':
                        if block.get('is_error'):
                            metrics['tools']['errors'] += 1
                            # NEW: Track errors for this specific agent
                            metrics['agents']['by_agent'][agent_id]['errors'] += 1
                            metrics['tools']['error_details'].append({
                                'tool_use_id': block.get('tool_use_id'),
                                'content': block.get('content', 'No error message'),
                                'from_subagent': is_subagent,
                            })
        
        # Extract final result metrics
        if msg_type == 'ResultMessage':
            metrics['costs']['total_cost_usd'] = msg_data.get('total_cost_usd', 0.0)
            metrics['costs']['num_turns'] = msg_data.get('num_turns', 0)
            metrics['costs']['duration_ms'] = msg_data.get('duration_ms', 0)
            
            usage = msg_data.get('usage', {})
            metrics['costs']['token_usage']['input_tokens'] = usage.get('input_tokens', 0)
            metrics['costs']['token_usage']['output_tokens'] = usage.get('output_tokens', 0)
            metrics['costs']['token_usage']['cache_creation_input_tokens'] = usage.get('cache_creation_input_tokens', 0)
            metrics['costs']['token_usage']['cache_read_input_tokens'] = usage.get('cache_read_input_tokens', 0)
            metrics['costs']['token_usage']['total_tokens'] = (
                metrics['costs']['token_usage']['input_tokens'] + 
                metrics['costs']['token_usage']['output_tokens']
            )
            
            metrics['costs']['cache_creation'] = usage.get('cache_creation', {})
            metrics['costs']['web_search_requests'] = usage.get('server_tool_use', {}).get('web_search_requests', 0)
    
    # Calculate derived metrics
    metrics['agents']['unique_subagent_instances'] = len(subagent_tool_use_ids)
    metrics['costs']['duration_seconds'] = metrics['costs']['duration_ms'] / 1000.0
    
    return metrics


def print_metrics_report(metrics: Dict[str, Any]) -> None:
    """Print a formatted report of the metrics."""
    
    print("\n" + "="*80)
    print("MULTI-AGENT SYSTEM METRICS REPORT")
    print("="*80)
    
    # Overview
    print("\n📊 OVERVIEW")
    print("-" * 80)
    print(f"Question: {metrics['overview']['question'][:100]}...")
    print(f"Expected Answer: {metrics['overview']['expected_answer']}")
    print(f"Received Answer: {metrics['overview']['received_answer'][:100]}...")
    print(f"Total Messages: {metrics['overview']['total_messages']}")
    
    # Agent Stats
    print("\n🤖 AGENT STATISTICS")
    print("-" * 80)
    print(f"Total Subagent Calls: {metrics['agents']['total_subagent_calls']}")
    print(f"Unique Subagent Instances: {metrics['agents']['unique_subagent_instances']}")
    print(f"Subagent Types: {', '.join(metrics['agents']['subagent_types'])}")
    print(f"\nModels Used:")
    for model, count in metrics['agents']['models_used'].items():
        print(f"  - {model}: {count} calls")
    
    # NEW: Per-agent breakdown
    print(f"\n📊 PER-AGENT METRICS")
    print("-" * 80)
    for agent_id, agent_metrics in metrics['agents']['by_agent'].items():
        agent_name = 'Lead Agent' if agent_id == 'lead_agent' else f'Subagent {agent_id[:8]}...'
        print(f"\n  {agent_name}:")
        print(f"    Model: {agent_metrics['model']}")
        print(f"    Messages: {agent_metrics['messages']}")
        print(f"    Tool Calls: {sum(agent_metrics['tool_calls'].values())}")
        print(f"    Errors: {agent_metrics['errors']}")
        if agent_metrics['tool_calls']:
            print(f"    Tools Used:")
            for tool, count in agent_metrics['tool_calls'].most_common():
                print(f"      - {tool}: {count}")
    
    # Message Breakdown
    print("\n📨 MESSAGE BREAKDOWN")
    print("-" * 80)
    print(f"Lead Agent Messages: {metrics['message_breakdown']['by_agent_level']['lead_agent']}")
    print(f"Subagent Messages: {metrics['message_breakdown']['by_agent_level']['subagent']}")
    print(f"\nBy Type:")
    for msg_type, count in metrics['message_breakdown']['by_type'].most_common():
        print(f"  - {msg_type}: {count}")
    
    # Tool Usage
    print("\n🔧 TOOL USAGE")
    print("-" * 80)
    print(f"Total Tool Calls: {metrics['tools']['total_tool_calls']}")
    print(f"Errors: {metrics['tools']['errors']}")
    print(f"\nBy Tool:")
    for tool, count in metrics['tools']['by_tool'].most_common():
        print(f"  - {tool}: {count} calls")
    
    print(f"\nBy Agent Level:")
    print(f"  Lead Agent:")
    for tool, count in metrics['tools']['by_agent_level']['lead_agent'].most_common():
        print(f"    - {tool}: {count}")
    print(f"  Subagents:")
    for tool, count in metrics['tools']['by_agent_level']['subagents'].most_common():
        print(f"    - {tool}: {count}")
    
    if metrics['tools']['error_details']:
        print(f"\n  Error Details:")
        for i, error in enumerate(metrics['tools']['error_details'], 1):
            source = "Subagent" if error['from_subagent'] else "Lead Agent"
            print(f"    {i}. [{source}] {error['content'][:80]}...")
    
    # Costs and Performance
    print("\n💰 COSTS & PERFORMANCE")
    print("-" * 80)
    print(f"Total Cost: ${metrics['costs']['total_cost_usd']:.4f}")
    print(f"Number of Turns: {metrics['costs']['num_turns']}")
    print(f"Duration: {metrics['costs']['duration_seconds']:.2f} seconds ({metrics['costs']['duration_ms']} ms)")
    print(f"\nToken Usage:")
    print(f"  - Input Tokens: {metrics['costs']['token_usage']['input_tokens']:,}")
    print(f"  - Output Tokens: {metrics['costs']['token_usage']['output_tokens']:,}")
    print(f"  - Total Tokens: {metrics['costs']['token_usage']['total_tokens']:,}")
    print(f"  - Cache Creation: {metrics['costs']['token_usage']['cache_creation_input_tokens']:,}")
    print(f"  - Cache Reads: {metrics['costs']['token_usage']['cache_read_input_tokens']:,}")
    print(f"\nWeb Search Requests: {metrics['costs']['web_search_requests']}")
    
    if metrics['costs']['cache_creation']:
        print(f"\nCache Creation Details:")
        for key, value in metrics['costs']['cache_creation'].items():
            print(f"  - {key}: {value:,}")
    
    print("\n" + "="*80 + "\n")


def get_metrics_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a condensed summary of key metrics.
    
    Args:
        data: The loaded JSON data
        
    Returns:
        Dictionary with key summary metrics
    """
    metrics = analyze_agent_metrics(data)
    
    return {
        'subagent_calls': metrics['agents']['total_subagent_calls'],
        'total_tool_calls': metrics['tools']['total_tool_calls'],
        'errors': metrics['tools']['errors'],
        'cost_usd': metrics['costs']['total_cost_usd'],
        'duration_seconds': metrics['costs']['duration_seconds'],
        'total_tokens': metrics['costs']['token_usage']['total_tokens'],
        'web_searches': metrics['costs']['web_search_requests'],
    }