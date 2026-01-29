# AI Writing Detector

An end-to-end Machine Learning pipeline designed to detect AI-generated text ("Slop").

This project implements a full lifecycle:
1.  **Scraping** verified human text (Wikipedia pre-2022).
2.  **Generating** adversarial AI text (GPT-5 Nano).
3.  **Training** a BERT classifier using LoRA (Low-Rank Adaptation).
4.  **Inference** for real-time detection.

##  Documentation
- [Pipeline Architecture](docs/ARCHITECTURE.md) - How the pieces fit together.
- [Data Scraping](docs/DATA_SCRAPING.md) - How we ensure human data is actually human.
- [Data Generation](docs/DATA_GENERATION.md) - The "Summarize-Then-Expand" technique.

##  Quick Start

### 1. Prerequisites
Ensure you have Python 3.10+ and the necessary dependencies installed.
```bash
pip install -r requirements.txt
# Or if using uv/poetry
uv sync
```

### 2. Generate Data
(Optional) If you don't have the dataset yet.
```bash
# Set your OpenAI Key
export OPEN_AI_KEY="sk-..."

# Run the generation pipeline
python AI-Writing-Detector/AI-Writing-Detector/data-generation/main.py
```

### 3. Train Model
Fine-tune the BERT model on your pairs of Human/AI text.
```bash
python AI-Writing-Detector/AI-Writing-Detector/training/src/main.py
```
This will save the model to `AI-Writing-Detector/AI-Writing-Detector/training/src/results/final_model`.

### 4. Inference
Test the model interactively.
```bash
python AI-Writing-Detector/AI-Writing-Detector/training/src/inference.py
```

##  Directory Structure
- `scraping/`: Scripts to fetch pre-2022 Wikipedia data.
- `data-generation/`: Scripts to act as the AI adversary.
- `training/`: Training loop (`main.py`) and Inference (`inference.py`).
- `data/`: Stores the `.jsonl` datasets.
