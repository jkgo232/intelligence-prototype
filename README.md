# Intelligence Prototype

CNN–LLM Pipeline for Signal Detection

This repository implements a toy-model of a pipeline for detecting, ranking, and interpreting narrowband drifting radio signals using a CNN anomaly detector followed by LLM-based reasoning and evaluation.



\##Overview

The pipeline consists of four main stages:

1. Tiling and Preprocessing: Large dynamic spectra are divided into fixed-size time–frequency tiles.
2. CNN Training with Synthetic Injections: A convolutional neural network is trained to identify anomalous tiles using injected drifting narrowband signals.
3. CNN Scoring of New Data: The trained CNN scores unseen tiles and ranks them by anomaly likelihood.
4. LLM-Based Evaluation and Interpretation: Top-ranked tiles are summarized via scalar features and evaluated by LLMs (chatgpt or local models).



--



\##Suggested Instructions:

1. download and store HDF5 raw data in \\data\\raw (see methods.md)
2. run get\_tiles.py -- this preprocesses data and saves tiles to \\data\\tiles\\
3. run test\_tiles.py -- plots a few random tiles for a sanity check - optional
4. Train CNN (skip this if you are using an already trained/saved CNN): run train\_cnn\_baseline\_injected\_signal.py -- this loads tiles, injects a simulated signal into a small fraction, measures features, splits data, trains and saves CNN, and outputs model diagnostics
5. run check\_features.py to generate feature plots
6. run score\_tiles.py using new tiles (i.e. different from the one's used for training - you will need to move the training tiles to \\data\\tiles\\tiles\_old and save the new data to \\data\\tiles or change the directories under 'config' that point to the location of your new tiles) this loads tiles, injects a simulated signal into a small fraction and outputs scored\_tiles.json and scored\_tiles\_inj.json
7. run check\_features.py again, if desired, to inspect new tiles
8. run plot\_tiles.py to see features vs. CNN score with injected and regular tiles in different colors - this aids in model evaluation and setting heuristics
9. run evaluate\_llm.py -- provides output from score\_tiles.py to a specified llm, classifies and provides an explanantion for 50 top scoring tiles.
   For more details, and changes needed based on your particular dataset, see comments in python scripts, below, and methods.md.
   Note: this is a toy model and has not yet been optimized for general use, or performance -- both will increase greatly with increased time investment



--



\##CNN Model Architecture:

Input: (128, 128, 1) normalized power tiles, 3× Conv2D layers with ReLU, Global Average Pooling, Sigmoid output



--



\##Training Strategy:

Binary classification: 0: background / noise, 1: injected drifting narrowband signal (or signal threshold i.e. 3 std)

~4–10% synthetic injections,

Early stopping on validation loss,

Metrics: Accuracy + AUC



--



\##Signal Injection:

Injected signals have: Random start time and frequency, Linear frequency drift, Narrow bandwidth (1–3 pixels)



--



\##CNN Scoring and Ranking:

The trained CNN is applied to unseen tiles using: python scripts/score\_tiles.py

Outputs:
scored\_tiles.json
→ Top-K tiles with CNN score + scalar features (no labels)

scored\_tiles\_inj.json
→ Same tiles with private injection metadata for validation

Only the public JSON is passed to LLMs.



--



\##Feature Extraction:

For each tile, the following scalar features are computed: Mean normalized power, Standard deviation, Maximum normalized power, Kurtosis, Spectral entropy



--



\##LLM Evaluation:

LLMs are prompted to:

1. Classify each tile as:

Likely technosignature

Likely RFI or Noise / artifact

2. Provide a short explanation

Supported LLMs:  ChatGPT (OpenAI API), Qwen (in progress)

Evaluation Metrics: Consistency (repeat prompts), Agreement with heuristic rules, Explanation completeness

Results are stored in: outputs/llm\_evaluation.json



--



\##LoRA Dataset Generation:

CNN-scored tiles ca be converted into a JSONL LoRA training dataset:

{"prompt": "...features...", "response": "...classification..."}

Injected tiles are labeled as likely technosignatures; others as RFI/noise for fine-tuning



--



\##Visualization:

Feature distributions and clustering are visualized with: Injected vs non-injected tiles, Feature vs CNN score, Color-coded injection recovery

