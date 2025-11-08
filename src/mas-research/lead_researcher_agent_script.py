from dotenv import load_dotenv
from utils.agent_visualizer import print_activity, visualize_conversation
from utils.message_serializer import save_messages, load_messages, serialize_message
from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions, ClaudeSDKClient, create_sdk_mcp_server, query, tool

from helpers import get_one_browsecomp_question_answer, get_result_from_messages

import asyncio
import os
import json
import pandas as pd
from datetime import datetime


haiku_45 = "claude-haiku-4-5-20251001"
sonnet_4 = "claude-sonnet-4-20250514"
sonnet_45 = "claude-sonnet-4-5"

dummy_model = haiku_45
model = sonnet_45
dummy_system_prompt = """
You are a lazy professional researcher. Your goal is to find the answer by delegating tasks to your subagents as much as possible. Use the 'researcher' subagent.
"""
dummy_question = "Who is leading the race in the NYC mayoral election in 2025? Use the researcher subagent to find out."
dummy_answer = "Zohran Mamdani"
dummy_tools = ["Read", "WebSearch"]


lead_researcher_prompt = ""
with open("prompts/research_lead_agent.md", "r") as f:
    lead_researcher_prompt = f.read()
    lead_researcher_prompt = lead_researcher_prompt.replace("{{.CurrentDate}}", datetime.now().strftime("%B %d, %Y"))


research_subagent_prompt = ""
with open("prompts/research_subagent.md", "r") as f:
    research_subagent_prompt = f.read()
    research_subagent_prompt = research_subagent_prompt.replace("{{.CurrentDate}}", datetime.now().strftime("%B %d, %Y"))


# @tool("submit_final_report", "Submit final research report", {"report": str})
# async def submit_final_report(args):
#     return {"content": [{"type": "text", "text": "Report submitted"}]}

# @tool("return_findings", "Return findings to lead agent", {"findings": str, "notes": str})
# async def return_findings(args):
#     return {"content": [{"type": "text", "text": "Findings returned"}]}

# # Create server with both
# server = create_sdk_mcp_server(
#     name="research-tools",
#     version="1.0.0", 
#     tools=[submit_final_report, return_findings]
# )

tools = ["WebSearch", "Read", "Task", "Bash"] ## , "submit_final_report", "return_findings"]



def save_result(result: dict, filecode: str = "browsecomp"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"search_results/{filecode}_result_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(result, f, indent=2)
    with open(f"{filecode}_current.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"✅ Saved result to {filename}")
    print(f"✅ Also updated {filecode}_current.json")

def export_to_md(result_text: str, filecode: str = "browsecomp"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"search_results/{filecode}_result_{timestamp}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(result_text.strip() + "\n")
    print(f"Markdown exported to {filename}")

async def run_one_search(model: str, system_prompt: str, subagent_prompt: str, question: str, tools: list, debug_verbose: bool = False):
    messages = []

    async with ClaudeSDKClient(
        options=ClaudeAgentOptions(
            model=model,
            # cwd="research_agent",
            system_prompt=system_prompt,
            allowed_tools=tools,
            max_turns=50,
            agents = {
            'research_subagent' : AgentDefinition(
                description="A fully capable researcher that can search the web.",
                prompt=subagent_prompt,
                model="haiku",
                tools=["WebSearch", "Read"]
                )
            }
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

    qa = get_one_browsecomp_question_answer(idx=0, print_question=True)
    question = qa['question']
    answer = qa['answer']

    # with open("prompts/solo_research_agent.md", "r") as f:
    #     PROMPT = f.read()
    #     PROMPT = PROMPT.replace("{{.CurrentDate}}", datetime.now().strftime("%B %d, %Y"))

    # question = dummy_question
    # answer = dummy_answer

    # messages = asyncio.run(run_one_search(model=dummy_model, system_prompt=lead_researcher_prompt, subagent_prompt=research_subagent_prompt, question=question, tools=tools, debug_verbose=True))
    messages = asyncio.run(run_one_search(model=dummy_model, system_prompt=lead_researcher_prompt, subagent_prompt=research_subagent_prompt, question=question, tools=tools, debug_verbose=True))
    visualize_conversation(messages)

    serialized = [serialize_message(msg) for msg in messages]
    response = get_result_from_messages(messages)

    result = {
        "question": question,
        "expected_answer": answer,
        "recieved_answer": response,
        "messages": serialized
    }

    save_result(result, filecode="multiagent_test")

    export_to_md(result_text=response, filecode="multiagent_test") # type: ignore

