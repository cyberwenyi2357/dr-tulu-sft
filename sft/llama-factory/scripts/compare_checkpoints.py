import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen3-8B"
CHECKPOINTS = {
    "checkpoint-100": "ckpts/qwen3-8b-clarify-classification-sft/checkpoint-100",
    "checkpoint-140": "ckpts/qwen3-8b-clarify-classification-sft/checkpoint-140",
    "checkpoint-250-final": "ckpts/qwen3-8b-clarify-classification-sft",
}
TEST_FILES = [
    "scripts/researchQA_sample50_fully_underspecified.jsonl",
    "scripts/researchQA_sample50_without_scope.jsonl",
    "scripts/researchQA_sample50_without_term.jsonl",
]

INSTRUCTION = (
    "You are an ambiguity checker for domain-specific user queries submitted to a deep research agent. "
    "Given a query, determine whether additional information from the user is needed for the query to "
    "be answered clearly and uniquely.\n"
    "Output yes if the query is underspecified and a clarification question is needed. Otherwise, "
    "output no. Output only a single token: `yes` or `no`."
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


def generate(model, query):
    messages = [{"role": "user", "content": f"{INSTRUCTION}\n{query}"}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


all_results = {}

for ckpt_name, ckpt_path in CHECKPOINTS.items():
    print(f"\n===== Loading {ckpt_name} =====")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model = PeftModel.from_pretrained(base_model, ckpt_path)
    model.eval()

    ckpt_results = {}
    for test_file in TEST_FILES:
        with open(test_file) as f:
            items = [json.loads(line) for line in f if line.strip()]

        preds = []
        for item in items:
            prediction = generate(model, item["query"]).strip()
            preds.append({"id": item.get("id"), "query": item["query"], "prediction": prediction})

        yes_count = sum(1 for p in preds if p["prediction"] == "Yes")
        no_count = sum(1 for p in preds if p["prediction"] == "No")
        other_count = len(preds) - yes_count - no_count
        print(f"[{ckpt_name}] {test_file}: total={len(preds)} Yes={yes_count} No={no_count} Other={other_count}")
        ckpt_results[test_file] = preds

    all_results[ckpt_name] = ckpt_results

    del model
    del base_model
    torch.cuda.empty_cache()

with open("scripts/checkpoint_comparison_predictions.json", "w") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print("\nSaved all predictions to scripts/checkpoint_comparison_predictions.json")
