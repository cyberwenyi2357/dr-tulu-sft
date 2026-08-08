import json
import os

from openai import OpenAI

MODEL = "gpt-5.4"
INPUT_FILE = "scripts/researchQA_sample50_without_term.jsonl"
INSTRUCTION_SOURCE = "data/augmented_clarify_classification_alpaca.json"
OUTPUT_FILE = INPUT_FILE.rsplit(".", 1)[0] + "_gpt_predictions_simplified_prompt.json"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

with open(INSTRUCTION_SOURCE) as f:
    data = json.load(f)
    INSTRUCTION = (data[0] if isinstance(data, list) else data)["instruction"]


def judge(query: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": f"{INSTRUCTION}\n{query}"}],
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    with open(INPUT_FILE) as f:
        items = [json.loads(line) for line in f if line.strip()]

    results = []
    for i, item in enumerate(items):
        query = item["query"]
        try:
            label = judge(query)
        except Exception as e:
            label = f"ERROR: {e}"

        print(f"[{i + 1}/{len(items)}] {query}\n{label}\n")
        results.append({"id": item.get("id"), "query": query, "label": label})

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(results)} results to {OUTPUT_FILE}")
