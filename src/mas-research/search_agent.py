from dotenv import load_dotenv
from utils.agent_visualizer import print_activity, visualize_conversation
from utils.message_serializer import save_messages, load_messages, serialize_message
from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions, ClaudeSDKClient, query

from helpers import get_one_browsecomp_question_answer

import asyncio
import os
import json
import pandas as pd
from datetime import datetime


haiku_45 = "claude-haiku-4-5-20251001"
sonnet_4 = "claude-sonnet-4-20250514"

dummy_model = haiku_45
dummy_system_prompt = """
Research the latest trends in AI agents and give me a brief summary
"""
dummy_question = "Who is leading the race in the NYC mayoral election in 2025?"
dummy_answer = "Zohran Mamdani"
dummy_tools = ["WebSearch", "Read", "complete_task"]


def save_result(result: dict, filecode: str = "browsecomp"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"search_results/{filecode}_result_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(result, f, indent=2)
    with open(f"{filecode}_current.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"✅ Saved result to {filename}")
    print(f"✅ Also updated {filecode}_current.json")


async def run_one_search(model: str, system_prompt: str, question: str, tools: list, debug_verbose: bool = False):
    messages = []
    max_messages = 50

    async with ClaudeSDKClient(
        options=ClaudeAgentOptions(
            model=model,
            # cwd="research_agent",
            system_prompt=system_prompt,
            allowed_tools=tools,
            max_turns=50
        )
    ) as research_agent:
        await research_agent.query(question)
        async for msg in research_agent.receive_response():
            print_activity(msg)
            messages.append(msg)
            
            if debug_verbose:
                print('\n', msg)
    
    return messages


if __name__ == "__main__":

    load_dotenv()

    # qa = get_one_browsecomp_question(idx=0, print_question=True)
    # question = qa['question']
    # answer = qa['answer']

    # with open("prompts/solo_research_agent.md", "r") as f:
    #     PROMPT = f.read()
    #     PROMPT = PROMPT.replace("{{.CurrentDate}}", datetime.now().strftime("%B %d, %Y"))

    question = dummy_question
    answer = dummy_answer

    messages = asyncio.run(run_one_search(model=dummy_model, system_prompt=dummy_system_prompt, question=question, tools=dummy_tools))
    visualize_conversation(messages)

    serialized = [serialize_message(msg) for msg in messages]
    result = {
        "question": question,
        "expected_answer": answer,
        "messages": serialized
    }
    save_result(result, filecode="test")

