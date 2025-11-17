from dotenv import load_dotenv # type: ignore
from utils.agent_visualizer import print_activity, visualize_conversation
from utils.message_serializer import save_messages, load_messages, serialize_message
from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions, ClaudeSDKClient, create_sdk_mcp_server, query, tool # type: ignore

from helpers import get_one_browsecomp_question_answer, get_result_from_messages, load_prompt, save_result, export_to_md
from evaluate_answer import evaluate_answer

import asyncio
import os
import json
import pandas as pd # type: ignore
from datetime import datetime
from typing import Any, Dict



# ## 'simple' mode: single agent with query() only. Deprecated.
# async def simple_agentic_query(model: str, system_prompt: str, question: str, tools: list, debug_verbose: bool = False):
#     options = ClaudeAgentOptions(
#         model=model,
#         system_prompt=system_prompt,
#         allowed_tools=tools,
#         disallowed_tools=['Task'],
#         max_turns=50,
#     )

#     async for message in query(
#         prompt=question,
#         options=options
#     ):
#         print(message)



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
                tools=["WebSearch", "WebFetch", "Read", "Bash"]
                )
            } if subagent_prompt and len(subagent_prompt) > 0 else None,
        )
    ) as research_agent:
        await research_agent.query(question)
        async for msg in research_agent.receive_response():
            print_activity(msg)
            messages.append(msg)
            
            if debug_verbose:
                print('\n', msg)
    
    return messages



# # ## Connection point from main_runner.py. All modes but eval ##
# Deprecated in favor of main_browsecomp_eval in run_eval.py   
# def run_via_config(config: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Small convenience wrapper to run the pipeline using a simple config dict.
#     - mode: "dummy" | "default"
#     """
#     mode = (config or {}).get("mode", "dummy")

#     debug_verbose = bool((config or {}).get("debug_verbose", False))
#     filecode = (config or {}).get("filecode", "misc")

#     model_to_use = config.get("model", "claude-haiku-4-5-20251001")
#     tools_to_use = config["tools"]


#     if mode == "dummy":
#         system_prompt_to_use = config["dummy_system_prompt"]
#         researcher_subagent_prompt_to_use = config["dummy_researcher_prompt"]
#         question = config["dummy_question"]
#         expected_answer = config["dummy_answer"]

#     else:
#         lead_researcher_prompt = load_prompt(config["system_prompt_filepath"]) 
#         system_prompt_to_use = lead_researcher_prompt

        
#         qa_idx = int(config.get("question_index", 0))
#         qa = get_one_browsecomp_question_answer(
#             idx=qa_idx, print_question=bool(config.get("print_question", False))
#         )
#         # Be robust to potential pandas Series types
#         question = str(qa["question"])  # type: ignore[arg-type]
#         expected_answer = str(qa["answer"])  # type: ignore[arg-type]


#         subagent_pathway = config.get("research_subagent_prompt_filepath", None)
#         researcher_subagent_prompt = load_prompt(subagent_pathway) if subagent_pathway else ""
#         researcher_subagent_prompt_to_use = researcher_subagent_prompt


#     messages = asyncio.run(
#         run_one_search(
#             model=model_to_use,
#             system_prompt=system_prompt_to_use,
#             subagent_prompt=researcher_subagent_prompt_to_use,
#             question=question,
#             tools=tools_to_use,
#             debug_verbose=debug_verbose,
#         )
#     )

#     visualize_conversation(messages)

#     serialized = [serialize_message(msg) for msg in messages]
#     response = get_result_from_messages(messages)


#     evaluation = evaluate_answer(
#         question=question,
#         correct_answer=expected_answer,
#         response=response if response else "No response received.",
#     )

#     result = {
#         "question": question,
#         "expected_answer": expected_answer,
#         "received_answer": response,
#         "evaluation": evaluation.get("evaluation", ""),
#         "grade": evaluation.get("grade", "no"),
#         "messages": serialized,
#     }



#     save_result(result, filecode=filecode)
#     export_to_md(result_text=response, filecode=filecode) # type: ignore

#     return result


# OLD CODE BELOW - FOR REFERENCE ONLY
# - --- IGNORE ---

# # if __name__ == "__main__":

# #     load_dotenv()

# #     qa = get_one_browsecomp_question_answer(idx=0, print_question=True)
# #     question = qa['question']
# #     answer = qa['answer']

# #     # with open("prompts/solo_research_agent.md", "r") as f:
# #     #     PROMPT = f.read()
# #     #     PROMPT = PROMPT.replace("{{.CurrentDate}}", datetime.now().strftime("%B %d, %Y"))

# #     question = dummy_question
# #     answer = dummy_answer

# #     messages = asyncio.run(run_one_search(model=dummy_model, system_prompt=dummy_system_prompt, subagent_prompt=research_subagent_prompt, question=question, tools=tools, debug_verbose=True))
# #     # messages = asyncio.run(run_one_search(model=dummy_model, system_prompt=lead_researcher_prompt, subagent_prompt=research_subagent_prompt, question=question, tools=tools, debug_verbose=True))
# #     visualize_conversation(messages)

# #     serialized = [serialize_message(msg) for msg in messages]
# #     response = get_result_from_messages(messages)

# #     result = {
# #         "question": question,
# #         "expected_answer": answer,
# #         "received_answer": response,
# #         "messages": serialized
# #     }

# #     save_result(result, filecode="multiagent_test")
# #     export_to_md(result_text=response, filecode="multiagent_test") # type: ignore

