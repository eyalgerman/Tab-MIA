# Extending Tab-MIA

This guide explains how to add new membership inference attack (MIA) methods and how to evaluate different language models using the Tab-MIA benchmark.

## Adding a New Attack Method

1. **Create a module**
   - Add your attack implementation under the `MIA` package. You can follow the structure of `mia_detection.py` as a reference.
   - Your module should expose a function that accepts a list of text inputs with their labels and returns a dictionary of metrics for each record.
2. **Integrate with the evaluation pipeline**
   - Update `main.py` (or `main_syn.py` for pretrained models) to import and call your attack function.
   - Make sure the metrics are written to disk in the same format as the default implementation so that `eval_2.py` can aggregate the results.
   - You can also modify the `inference()` function in `MIA/mia_detection.py` to compute your attack directly. Store the new metric in the returned dictionary as `pred["<attack name>"]` so it appears in the output CSV.
3. **Document the new attack**
   - Provide a short description and usage example in this file or in the project README so other users know how to invoke it.

## Using a Different Language Model

1. **Select the model**
   - Pass the model name or path via the `--target_model` argument when running `main.py` or `main_syn.py`.
   - Any model hosted on the Hugging Face Hub can be used (e.g., `mistralai/Mistral-7B-v0.1`).
2. **Fine-tuning or loading**
   - The script checks for previously fine-tuned models in `output_dir/Models`. If found, it reuses them when `--use_existing model` or `--use_existing all` is set.
   - If no fine-tuned model exists, `finetune_LLM.py` will launch a QLoRA fine-tuning run and save the merged model.
3. **Run MIA detection**
   - After fine-tuning (or when using a pretrained model), the pipeline calls `MIA/mia_detection.py` to compute attack metrics.
   - The results are saved under `output_dir/results/<model>/<dataset>`.

## Using a Custom CSV Dataset

1. **Provide a CSV file**
   - Use the `--data` argument with the path to your CSV when running `main.py`.
   - Choose the table serialization format via `--table_encoding` (e.g., `json`, `markdown`).
2. **Automatic conversion**
   - The script splits the CSV into member and non-member tables using `process_data/process_csv_file.py`.
   - Converted files and a `.jsonl` version are stored under `output_dir/Datasets/<csv_name>/`.
3. **Run on a pretrained model only**
   - For evaluation without fine-tuning, pass the CSV path to `main_syn.py` instead.

## Example Workflow

```bash
python main.py --data tabMIA_adult \
                --target_model mistralai/Mistral-7B-v0.1 \
                --output_dir ./results \
                --num_epochs 3 \
                --use_existing all \
                --table_encoding json
```

To test a custom attack `my_attack.py` located in `MIA/`, import it in `main.py` and call it in place of or in addition to `mia_detection.main`.

For quick experiments without fine-tuning, use `main_syn.py` with a JSONL file of your own data:

```bash
python main_syn.py --data path/to/data.jsonl \
                   --target_model mistralai/Mistral-7B-v0.1 \
                   --output_dir ./results \
                   --use_existing all \
                   --table_encoding json
```
