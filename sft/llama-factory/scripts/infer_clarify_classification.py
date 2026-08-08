import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "ckpts/qwen3-8b-clarify-classification-sft-merged"
INPUT_FILE = "data/query_with_positive_diff_merged_diff_gt_0.1.jsonl"
OUTPUT_FILE = "data/query_with_positive_diff_merged_diff_gt_0.1_clarify_classification_predictions.json"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0")
model.eval()

INSTRUCTION = (
    "You are an ambiguity checker for domain-specific user queries submitted to a deep research agent. "
    "Given a query, determine whether additional information from the user is needed for the query to "
    "be answered clearly and uniquely.\n"
    "Output yes if the query is underspecified and a clarification question is needed. Otherwise, "
    "output no. Output only a single token: `yes` or `no`."
)

def generate(query):
    messages = [{"role": "user", "content": f"{INSTRUCTION}\n{query}"}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

if __name__ == "__main__":
    with open(INPUT_FILE) as f:
        items = [json.loads(line) for line in f if line.strip()]

    results = []
    for i, item in enumerate(items):
        query = item["query"]
        prediction = generate(query)
        print(f"[{i + 1}/{len(items)}] {query}\n{prediction}\n")
        results.append({"id": item.get("id"), "query": query, "prediction": prediction})

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
