from datetime import datetime
import os
import warnings
import pandas as pd
import json

from utils.message_serializer import serialize_message



## Get one question and answer from the BrowseComp test set. 
## Output: { "question": ..., "answer": ... }
def get_one_browsecomp_question_answer(idx=0, print_question=False, filepath="data/browse_comp_test.csv"):

    bc = pd.read_csv(filepath).head()

    if idx >= len(bc):
        raise ValueError(f"Index {idx} out of range for BrowseComp test set with {len(bc)} entries.")
    
    question = bc.iloc[idx]['problem']
    answer = bc.iloc[idx]['answer']
    
    if print_question: print(f"--Question: {question}\nExpected Answer: {answer}\n")
    
    res = {
        "question": question,
        "answer": answer
    }
    return res

def get_browsecomp_result(filepath: str):
    with open(filepath, "r") as f:
        data = json.load(f)
        return data

## Get all questions and answers from the BrowseComp test set as JSON
## Output: [ { "question": ..., "answer": ... }, ... ]
def get_browsecomp_qas(filepath, num_rows=None, offset=4):
    bc = pd.read_csv(filepath)

    if offset >= len(bc):
        return []

    if num_rows is not None:
        bc = bc.iloc[offset:offset + num_rows]
    else:
        bc = bc.iloc[offset:]

    questions_and_answers = []
    for _, row in bc.iterrows():
        questions_and_answers.append({
            "question": row['problem'],
            "answer": row['answer']
        })

    return questions_and_answers
    


def get_final_result_from_saved_messages(filepath):

    messages = []
    with open(filepath, "r") as f:
        messages = json.load(f)

    res_msg = messages[-1]
    
    if 'result' in res_msg['data']:
        if res_msg['data']['result']:
            return res_msg['data']['result']
        
    print(f"❌ Format Error: No result found in the last message")
    return None

def load_messages_from_json(filepath: str):
    with open(filepath, "r") as f:
        data = json.load(f)
        return data
    
def get_result_from_messages(messages = None, filepath = None) -> str | None:
    
    if messages is None:
        if filepath:
            messages = load_messages_from_json(filepath)
        else:
            print("No messages or filepath provided.")
            return None
        
    if type(messages) != list:
        warnings.warn("Messages is not a list.")
    
    elif type(messages[0]) != dict:
        warnings.warn("Messages do not appear to be serialized dictionaries. Attempting to serialize...")
        messages = [serialize_message(msg) for msg in messages]
    
    for msg in messages:
        if msg.get('type') == "ResultMessage":
            result = msg.get('data', {}).get('result', None)
            if result:
                return result
    return None


def save_result(result: dict, filecode: str = "browsecomp", num = ""):
    # os.makedirs(f"results/{filecode}", exist_ok=True)
    timestamp = datetime.now().strftime("%b-%d-%H-%M")
    filename = f"{timestamp}"
    os.makedirs(f"results/{filecode}/{filename}", exist_ok=True)

    with open(f"results/{filecode}/{filename}/result{str(num)}.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(f"results/{filecode}/current.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"✅ Saved result to {filename}")
    print(f"✅ Also updated {filecode}_current.json")

def export_to_md(result_text: str, filecode: str = "browsecomp", num = ""):
    timestamp = datetime.now().strftime("%b-%d-%H-%M")
    filename = f"{timestamp}"

    os.makedirs(f"results/{filecode}/{filename}", exist_ok=True)
    with open(f"results/{filecode}/{filename}/report{str(num)}.md", "w", encoding="utf-8") as f:
        f.write(result_text.strip() + "\n")
    print(f"Markdown exported to {filename}")


# Load a prompt from a markdown file and replace date placeholders
def load_prompt(filepath: str) -> str:
    with open(filepath, "r") as f:
        researcher_prompt = f.read()
        researcher_prompt = researcher_prompt.replace("{{.CurrentDate}}", datetime.now().strftime("%B %d, %Y"))
        return researcher_prompt
    

if __name__ == "__main__":
    data = get_one_browsecomp_question_answer()
    print(data)

