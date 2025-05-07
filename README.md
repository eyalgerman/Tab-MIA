# Tab-MIA: Benchmarking Membership Inference Attacks on LLMs for Tabular Data

This repository contains the official code and datasets for the paper:

---

## 📄 Project Overview

We present **Tabular-MIA**, the first benchmark designed to evaluate **membership inference attacks (MIAs)** on large language models (LLMs) trained on **tabular data**. We systematically explore various **table encodings**, **model architectures**, and **attack methods**, introducing a unified evaluation framework for analyzing privacy risks in tabular LLM training.

Key contributions include:
- A suite of datasets with consistent train/test/synthetic splits
- Encoding schemes: JSON, HTML, Markdown, Key-Value formats, Key-Is-Value, Line-separated text
- Support for fine-tuning LLMs with QLoRA
- Comprehensive MIA evaluation pipeline with AUROC, TPR@FPR, and robustness tests

---

## 📦 Installation

```bash
git clone https://github.com/your-org/tabular-mia.git
cd tabular-mia
conda create -n tabular-mia-env python=3.11 -y
conda activate tabular-mia-env

# Install dependencies
pip install -r requirements.txt
```
---

## 📊 Datasets

The datasets can be found and downloaded from Hugging Face: [Tab-MIA](https://huggingface.co/datasets/germane/Tab-MIA)

We provide formatted datasets for:

- Adult Income
- California Housing
- TabFact
- WikiTableQuestions (WTQ)
- WikiSQL

Each dataset is provided in six different encodings:
- JSON - Serializes the table as a JSON list of records.
- HTML - Converts the table into an HTML `<table>` element.
- Markdown - Formats the table as a Markdown table.
- Key-Value Pairs - Each cell is represented as a "Key: Value" pair.
- Key-Is-Value - Similar to Key-Value, but in a natural language sentence format.
- Line-separated - Outputs each row as a comma-separated line of text.


Each file includes an indicator specifying whether each record is a member (part of the training set) or a non-member.

Directory structure:
```
datasets/
  └── adult/
      ├── adult_format_json.jsonl
      ├── adult_format_html.jsonl
      └── ...
```

---

## 🚀 Running the Code

To run the full pipeline, use the main script:

Parameter explanations:
- `--target_model`: Name or path of the base LLM to fine-tune (e.g., `mistralai/Mistral-7B-v0.1` or any other model available on Hugging Face)).
- `--output_dir`: Directory where models, logs, and result files will be saved.
- `--num_epochs`: Number of epochs for QLoRA fine-tuning.
- `--use_existing`: Whether to reuse previously generated data/models (`all`, `data`, or `model`).
- `--table_encoding`: The encoding format used to serialize tables. Supported values are:
  - `json` – JSON list of row dictionaries
  - `html` - HTML `<table>` format
  - `markdown` – Markdown table syntax
  - `key-value-pair` – "Key is Value" per cell
  - `key-is-value` – Same as above, in natural sentence form
  - `line-sep` – Line-separated, comma-separated rows

### Dataset Input Modes 
- **Run Tab-MIA**: When `--data` starts with `tabMIA_`, the data is automatically fetched from Hugging Face datasets (e.g., `tabMIA_adult`).
- **Create Tab-MIA Dataset**:
  - **Long-Context Tables**: To create a Tab-MIA dataset for long-context tables, use `--data` with a path to a CSV file. The code will process the CSV into text chunks based on the selected encoding if the encoded version does not already exist.
  - **Short-Context Tables**: For short-context tables, the dataset name should match one of the short-context datasets (`wtq`, `wikisql` or `tabfact` only). The code will generate the .jsonl files in the specified encoding if they do not already exist.
- **Pretrained Model**: When run `main_syn.py` file,`--data` needs to be a path to a JSONL file, the code will use the provided data for MIA detection without fine-tuning."
### 

### Run Tab-MIA with fine-tuning and MIA detection
```bash
python main.py --data tabMIA_adult                
                --target_model mistralai/Mistral-7B-v0.1 \
                --output_dir results/ \            
                --num_epochs 3 \
                --use_existing all \
                --table_encoding json               
```
The script handles:
- Preprocessing the data for each encoding
- Fine-tuning with QLoRA (or reusing a previous model)
- Running MIA detection.


### Run Tab-MIA with MIA detection only on pretrained model
```bash
python main_syn.py --data <path_to_JSONL_file> \
                    --target_model mistralai/Mistral-7B-v0.1 \
                    --output_dir results/ \
                    --use_existing all \
                    --table_encoding json
```


[//]: # (## 📚 Citation)

[//]: # ()
[//]: # (If you use this work, please cite:)

[//]: # ()
[//]: # (```bibtex)

[//]: # (@article{german2025tabularmia,)

[//]: # (  title={Tabular-MIA: Benchmarking Membership Inference Attacks on LLMs for Tabular Data},)

[//]: # (  author={German, Eyal and Shechner, Daniel and Shabtai, Asaf},)

[//]: # (  journal={NeurIPS},)

[//]: # (  year={2025})

[//]: # (})

[//]: # (```)

[//]: # ()
[//]: # (---)

## 🛡 License

This project is licensed under the MIT License. 