import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "ckpts/qwen3-8B-sft-final_v1-merged"
INPUT_FILES = [
    "scripts/researchQA_sample50_fully_underspecified.jsonl",
    "scripts/researchQA_sample50_without_scope.jsonl",
    "scripts/researchQA_sample50_without_term.jsonl",
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0")
model.eval()

INSTRUCTION = (
    'Determine whether the following query can be answered uniquely and clearly. '
    'Return a JSON object with exactly three fields: "label", "missing_information", and '
    '"follow_up_questions". Set "label" to "answerable" if the query is sufficiently clear and '
    'allows a specific, unique explanation or answer to be inferred. Set "label" to '
    '"needs clarification" if the query is too vague, lacks context, or requires the user to '
    'provide more information before it can be answered. If "label" is "needs clarification", '
    'set "missing_information" to a list of the missing information needed to answer the query '
    'and set "follow_up_questions" to a list of concise clarifying questions to ask the user. '
    'If "label" is "answerable", set "missing_information" to an empty list and '
    '"follow_up_questions" to an empty list.'
)

def generate(query):
    messages = [{"role": "user", "content": f"{INSTRUCTION}\n{query}"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

if __name__ == "__main__":
    for input_file in INPUT_FILES:
        with open(input_file) as f:
            items = [json.loads(line) for line in f if line.strip()]

        results = []
        for i, item in enumerate(items):
            query = item["query"]
            prediction = generate(query)
            print(f"[{input_file}] [{i + 1}/{len(items)}] {query}\n{prediction}\n")
            results.append({"id": item.get("id"), "query": query, "prediction": prediction})

        output_file = input_file.replace(".jsonl", "_v1_predictions.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
