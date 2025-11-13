from typing import Any, Dict
from dotenv import load_dotenv # type: ignore
import warnings
from evaluate_answer import evaluate_answer
from researcher_agent_script import run_one_search
from utils.agent_visualizer import print_activity, visualize_conversation
from utils.message_serializer import save_messages, load_messages, serialize_message
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, query # type: ignore

from helpers import get_browsecomp_qas, get_one_browsecomp_question_answer, get_result_from_messages, load_prompt, save_result, export_to_md

import asyncio
import os
import json
import pandas as pd
from datetime import datetime
import time
import argparse

browsecomp_qa_pd_filepath = "data/browse_comp_test.csv"


# load_dotenv()

# TESTING = False

# haiku_45 = "claude-haiku-4-5-20251001"
# sonnet_4 = "claude-sonnet-4-20250514"

# filecode = "browsecomp"
# tools = ["WebSearch", "Read", "Task", "Bash"]
# system_prompt = ""
# with open("prompts/solo_research_agent.md", "r") as f:
#     system_prompt = f.read()
#     system_prompt = system_prompt.replace("{{.CurrentDate}}", datetime.now().strftime("%B %d, %Y"))


# dummy_system_prompt = """
# Quickly and succinctly research the provided question using available tools, and provide a concise answer.
# """
# dummy_question = "Who won the race for the NYC mayoral election in 2025?"
# dummy_answer = "Zohran Mamdani"
# dummy_incorrect_answer = "Benjamin Grayzel"
# dummy_question_2 = "Is the US government currently shut down?"
# dummy_answer_2 = "Yes"
# dummy_tools = ["WebSearch", "Read"]

# dummy_3loop = [
#     {
#         "question": dummy_question,
#         "answer": dummy_answer
#     },
#     {
#         "question": dummy_question,
#         "answer": dummy_incorrect_answer
#     },
#     {
#         "question": dummy_question_2,
#         "answer": dummy_answer_2
#     }
# ]

# if TESTING:
#         tools = dummy_tools
#         system_prompt = dummy_system_prompt
#         filecode = "test"


# # if hasattr(msg, "model_dump"):
# #     print(msg.model_dump())


# def main():
#     print(filecode)
#     print(" \n\n=== Running Research Agent ===\n")

#     print("STEP 1 Fetching one BrowseComp question and answer...\n")
#     qa = get_one_browsecomp_question_answer(idx=0, print_question=(not TESTING))

#     question = qa['question'] if not TESTING else dummy_question
#     answer = qa['answer'] if not TESTING else dummy_answer

#     print("STEP 2 Running research agent...\n")
#     messages = asyncio.run(
#         run_one_search(
#             model=haiku_45,
#             system_prompt=system_prompt,
#             subagent_prompt="",
#             question=question,
#             tools=tools,
#             debug_verbose=False
#         ))
    
#     visualize_conversation(messages)

#     print("STEP 3 Evaluating result...\n")
#     messages = [serialize_message(msg) for msg in messages]
#     response = get_result_from_messages(messages)


#     assert isinstance(response, str)
#     grade = evaluate_answer(
#         question=question,
#         correct_answer=answer,
#         response=response
#         )
    
#     print(f"Grading result: \n_________________\n{grade}")

#     result = {
#         "question": question,
#         "expected_answer": answer,
#         "recieved_answer": response,
#         "grade": grade,
#         "messages": messages
#     }
#     print(result['question'],result['expected_answer'], result['recieved_answer'], result['grade'], len(result['messages']))

#     save_result(result, filecode=filecode)


# def main_loop_n(n = 3):
#     print(filecode)
#     print(" \n\n=== Running Research Agent ===\n")

#     qas = get_browsecomp_qas(browsecomp_qa_pd_filepath, num_rows=n) if not TESTING else dummy_3loop
#     print(f"Loaded {len(qas)} questions and answers.\n")

#     results = []
#     for qa in qas:

#         question = qa['question']
#         answer = qa['answer']

#         messages = asyncio.run(
#             run_one_search(
#                 model=haiku_45,
#                 system_prompt=system_prompt,
#                 subagent_prompt="",
#                 question=question,
#                 tools=tools,
#                 debug_verbose=False
#             ))
        
#         visualize_conversation(messages)

#         messages = [serialize_message(msg) for msg in messages]
#         response = get_result_from_messages(messages)


#         assert isinstance(response, str)
#         grade = evaluate_answer(
#             question=question,
#             correct_answer=answer,
#             response=response
#             )
        

#         result = {
#             "question": question,
#             "expected_answer": answer,
#             "recieved_answer": response,
#             "grade": grade,
#             "messages": messages
#         }
#         print(f"Question: {result['question']}")
#         print(f"Expected Answer: {result['expected_answer']}")
#         print(f"Received Answer: {result['recieved_answer']}")
#         print(f"Grade: {result['grade']}")
#         print(f"Messages Count: {len(result['messages'])}\n")

#         results.append(result)
#         save_result(result, filecode=filecode+"_mainloop"+str(n)+"")


def dummy_run_one_search(model: str, system_prompt: str, subagent_prompt: str, question: str, tools: list, debug_verbose: bool = False):
    with open("results/agent/current.json", "r") as f:
        results = json.load(f)
        messages = results['messages']
        return messages


async def run_search_wrapper(model: str, system_prompt: str, subagent_prompt: str, question: str, answer: str, tools: list, debug_verbose: bool = False):

    print(f"\nCOOKING Question: {question}, with expected answer: {answer}, is COOKING.\n")


    messages = await run_one_search(
            model=model,
            system_prompt=system_prompt,
            subagent_prompt=subagent_prompt,
            question=question,
            tools=tools,
            debug_verbose=debug_verbose
        )

    # messages = dummy_run_one_search( # HERE FOR TESTING TODO CHANGE DELETE
    #         model=model,
    #         system_prompt=system_prompt,
    #         subagent_prompt=subagent_prompt,
    #         question=question,
    #         tools=tools,
    #         debug_verbose=debug_verbose
    #     )
    
    print(f"\nCOMPLETE Question: {question}, with expected answer: {answer}, is COMPLETE.\n")
    # visualize_conversation(messages)
    messages = [serialize_message(msg) for msg in messages]
    response = get_result_from_messages(messages)

    if not isinstance(response, str):
        warnings.warn("Response is not a string!")
        response = "<empty>"

    evaluation = evaluate_answer(
        question=question,
        correct_answer=answer,
        response=response
        )
    
    result = {
        "question": question,
        "expected_answer": answer,
        "recieved_answer": response,
        "evaluation": evaluation.get("evaluation", ""),
        "grade": evaluation.get("grade", "no"),
        "messages": messages
    }

    return result




async def main_browsecomp_eval(config: Dict[str, Any]):

    print(" \n\n=== Running Research Agent ===\n")

    debug_verbose = bool((config or {}).get("debug_verbose", False))
    filecode = (config or {}).get("filecode", "misc")
    model_to_use = config.get("model", "claude-haiku-4-5-20251001")
    tools_to_use = config["tools"]

    lead_researcher_prompt = load_prompt(config["system_prompt_filepath"]) 
    system_prompt_to_use = lead_researcher_prompt

    subagent_pathway = config.get("research_subagent_prompt_filepath", None)
    researcher_subagent_prompt = load_prompt(subagent_pathway) if subagent_pathway else ""
    researcher_subagent_prompt_to_use = researcher_subagent_prompt

    # NEW: bounded concurrency
    max_concurrency = int(config.get("max_concurrency", 5))
    sem = asyncio.Semaphore(max_concurrency)

    offset = config.get("offset", 0)
    qas = get_browsecomp_qas(browsecomp_qa_pd_filepath, config.get("num_questions", 1), offset)
    print(f"Loaded {len(qas)} questions and answers.\n")


    async def sem_wrapper(qa):
        async with sem:
            return await run_search_wrapper(
                model = model_to_use, 
                system_prompt = system_prompt_to_use, 
                subagent_prompt = researcher_subagent_prompt_to_use, 
                question = qa['question'], 
                answer = qa['answer'], 
                tools = tools_to_use, 
                debug_verbose = debug_verbose
            )

    tasks = [asyncio.create_task(sem_wrapper(qa)) for qa in qas]

    results = await asyncio.gather(*tasks)

    for idx, result in enumerate(results):
        print("\n----- Result -----\n")
        print(f"Question: {result['question']}")
        print(f"Expected Answer: {result['expected_answer']}")
        print(f"Received Answer: {result['recieved_answer']}"if len(result['recieved_answer'])<100 else f"Received Answer: {result['recieved_answer'][:100]}...[truncated]")
        print(f"Grade: {result['grade']}")
        print(f"Messages Count: {len(result['messages'])}\n\n")

        save_result(result, filecode=filecode, num=str(idx+offset))
        export_to_md(result_text=result['recieved_answer'], filecode=filecode, num=str(idx+offset)) # type: ignore
        time.sleep(1.2)  # To ensure unique filenames formerly, idk why now
    return results


# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--test", action="store_true", help="Run in test mode")
#     args = parser.parse_args()
#     TESTING = args.test
#     print(f"TESTING mode: {TESTING}")

#     if TESTING:
#         tools = dummy_tools
#         system_prompt = dummy_system_prompt
#         filecode = "test"

#     qas = get_browsecomp_qas(browsecomp_qa_pd_filepath, num_rows=3) if not TESTING else dummy_3loop
#     asyncio.run(main_run_all_qs(qas))
#     # main_loop_n(n=3)







