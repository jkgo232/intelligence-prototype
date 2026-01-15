import json
import random

'''
uses CNN scores and features as output by score_tiles.py to generate a file that can be used for fine-tuning of an LLM

Need to adjust prompts and explanations as in evaluate_llm.py
'''

# config

INPUT_JSON = "../outputs/scored_tiles_inj.json"
OUTPUT_JSONL = "../outputs/lora_training.jsonl"

# adjust as needed based on data

MAX_BACKGROUND_RATIO = 10   # at most N background tiles per injected tile
SEED = 42

random.seed(SEED)

# define functions

def build_prompt(tile):
    f = tile["features"]

    prompt = f"""I would like you to analyze candidate radio signals.

Candidate properties:
CNN anomaly score: {tile['cnn_score']:.2f}
Mean normalized power: {f['mean']:.2f}
Standard deviation: {f['std']:.2f}
Maximum normalized power: {f['max']:.2f}
Kurtosis: {f['kurtosis']:.2f}
Spectral entropy: {f['entropy']:.2f}

Tasks:
1. Classify the signal as one of:
   - Likely technosignature
   - Likely RFI
   - Noise / artifact

2. Briefly explain your reasoning.

Please respond in the following format:
Classification: <label>
Explanation: <text>
"""
    return prompt.strip()


def build_response(tile):
    f = tile["features"]

    if tile["injected"]:
        label = "Likely technosignature"
        explanation = (
            "The signal shows strong narrowband power with high kurtosis and low "
            "spectral entropy, consistent with a coherent, non-noise source. "
            "The elevated CNN anomaly score further supports its classification "
            "as a potential technosignature."
        )
    else:
        label = "Likely RFI"
        explanation = (
            "The signal lacks the combination of low entropy and extreme kurtosis "
            "expected for a coherent narrowband source. Its properties are more "
            "consistent with terrestrial interference or noise fluctuations."
        )

    return f"""Classification: {label}
Explanation: {explanation}"""

# load data

with open(INPUT_JSON) as f:
    data = json.load(f)

injected = [d for d in data if d["injected"]]
background = [d for d in data if not d["injected"]]

print(f"Injected tiles: {len(injected)}")
print(f"Background tiles: {len(background)}")

# fix ratios -- adjust later

if injected:
    max_bg = MAX_BACKGROUND_RATIO * len(injected)
    background = random.sample(background, min(len(background), max_bg))

dataset = injected + background
random.shuffle(dataset)

# ============================== Write JSONL ==============================

with open(OUTPUT_JSONL, "w") as f:
    for tile in dataset:
        record = {
            "prompt": build_prompt(tile),
            "response": build_response(tile),
        }
        f.write(json.dumps(record) + "\n")

print(f"Wrote {len(dataset)} LoRA training examples to {OUTPUT_JSONL}")
