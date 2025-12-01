

import re
from helpers import get_browsecomp_result, get_final_result_from_saved_messages, get_one_browsecomp_question_answer, get_result_from_messages, load_messages_from_json

from browsecomp_openai.samplers import ChatCompletionSampler

from dotenv import load_dotenv  # type: ignore
import os

load_dotenv()

# BrowseComp evaluation implementation
QUERY_TEMPLATE = """
{Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

correctness_score: Put an integer from 0 to 10 indicating how close the extracted_final_answer is to the [correct_answer], where 0 = completely wrong or missing, 5 = partially correct with major errors or omissions, and 10 = fully correct with at most trivial differences.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.

""".strip()

CHOICE_STRINGS = ["yes", "no"]

GRADER_MODEL_NAME="gpt-5-mini" # "gpt-4"
TEST_JSON_PATH = "browsecomp_test.json" # TODO update to actual test file path

## This is the main evaluation function.
## Takes in a question, correct answer, and model response, and returns "yes" or "no".
def evaluate_answer(question: str, correct_answer: str, response: str) -> dict:
    grader_prompt = GRADER_TEMPLATE.format(
        question=question,
        correct_answer=correct_answer,
        response=response,
    )

    # Set up grader model with custom API parameters
    grader_model = ChatCompletionSampler(
        model=GRADER_MODEL_NAME,
    )

    prompt_messages = [
        grader_model._pack_message(content=grader_prompt, role="user")
    ]
    
    sampler_response = grader_model(prompt_messages)
    grading_response = sampler_response.response_text

    # Parse fields from grader response
    extracted_match = re.search(
        r"extracted[_\s-]*final[_\s-]*answer\s*:\s*(.+)",
        grading_response,
        flags=re.IGNORECASE,
    )
    extracted_final_answer = None
    if extracted_match:
        extracted_final_answer = extracted_match.group(1).strip().splitlines()[0].strip()
        # Normalize common placeholders and strip surrounding quotes
        if extracted_final_answer.lower() in {"none", "null", "n/a"}:
            extracted_final_answer = None
        elif (extracted_final_answer.startswith('"') and extracted_final_answer.endswith('"')) or (
            extracted_final_answer.startswith("'") and extracted_final_answer.endswith("'")
        ):
            extracted_final_answer = extracted_final_answer[1:-1].strip()

    match = re.search(r"correct:\s*(yes|no)", grading_response, flags=re.IGNORECASE)
    correctness_match = re.search(r"correctness_score:\s*([0-9]+(?:\.[0-9]+)?)", grading_response, flags=re.IGNORECASE)
    confidence_match = re.search(r"confidence:\s*([0-9]+(?:\.[0-9]+)?)\s*%?", grading_response, flags=re.IGNORECASE)

    eval = {
        "evaluation": grading_response,
        "grade": match.group(1).lower() if match else "no",
        "correctness": float(correctness_match.group(1)) if correctness_match else None,
        "confidence": float(confidence_match.group(1)) if confidence_match else None,
        "extracted_final_answer": extracted_final_answer,
    }

    return eval


def test_evaluate_answer():
    sample = get_one_browsecomp_question_answer(idx=0, print_question=False)
    question = str(sample["question"])
    correct_answer = str(sample["answer"])

    response = get_final_result_from_saved_messages(filepath=TEST_JSON_PATH)

    assert isinstance(response, str)

    print(f"Question: \n_________________\n{question}\n")
    print(f"Expected Answer: \n_________________\n{correct_answer}\n")
    print(f"Model response: \n_________________\n{response}\n") 

    grade_result = evaluate_answer(
        question=question,
        correct_answer=correct_answer,
        response=response
    )

    print(f"Grading result: \n_________________\n{grade_result}")

def test_evaluate_answer_from_result():
    result = get_browsecomp_result(filepath="test_current.json")

    question = result["question"]
    correct_answer = result["expected_answer"]
    messages = result["messages"]

    response = get_result_from_messages(messages)

    print(f"Question: \n_________________\n{question}\n")
    print(f"Expected Answer: \n_________________\n{correct_answer}\n")
    print(f"Model response: \n_________________\n{response}\n") 

    assert isinstance(response, str)

    grade_result = evaluate_answer(
        question=question,
        correct_answer=correct_answer,
        response=response
    )

    print(f"Grading result: \n_________________\n{grade_result}")

if __name__ == "__main__":
    # test_evaluate_answer_from_result()
    pass





