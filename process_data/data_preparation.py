import csv
import json
import os
import random
import sys

import pandas as pd
from datasets import load_dataset

from process_data import tables_encoder

result = []
TOP_K = 10000

def convert2df(table_dict):
    """
    Converts a dictionary with 'header' and 'rows' keys into a pandas DataFrame.

    Args:
        table_dict (dict): Dictionary with 'header' (list of column names) and 'rows' (list of rows).

    Returns:
        pandas.DataFrame: The resulting DataFrame with all values converted to strings.
    """
    columns = table_dict['header']
    data = table_dict['rows']
    df = pd.DataFrame(data, columns=columns)
    df = df.astype(str)
    return df


def convertdf2string(df):
    """
    Converts a pandas DataFrame to a CSV string without the index.

    Args:
        df (pandas.DataFrame): The DataFrame to convert.

    Returns:
        str: CSV-formatted string.
    """
    df = df.to_csv(index=False).strip('\n').split('\n')
    df_string = '\n'.join(df)
    return df_string

def convert_to_df(table_dict, data_mode):
    """
    Converts a table dictionary or text format to a DataFrame depending on dataset mode.

    Args:
        table_dict (dict or str): Table data structure.
        data_mode (str): Dataset type (e.g., "tab_fact", "WTQ").

    Returns:
        pandas.DataFrame: Converted DataFrame.
    """
    if data_mode == "tab_fact":
        lines = table_dict.strip().split('\n')
        columns = lines[0].split('#')
        data_rows = [line.split('#') for line in lines[1:]]
        df = pd.DataFrame(data_rows, columns=columns)
        return df
    columns = table_dict['header']
    data = table_dict['rows']
    df = pd.DataFrame(data, columns=columns)
    return df

def convert_df_to_string(df):
    """Converts a DataFrame to a CSV string with commas as delimiters."""
    return df.to_csv(index=False, sep=',')


def csv_to_jsonl(member_csv_path, non_member_csv_path, output_jsonl_path):
    """
    Merges member and non-member CSVs into a labeled JSONL file for MIA.

    Args:
        member_csv_path (str): Path to the CSV file containing member records.
        non_member_csv_path (str): Path to the CSV file containing non-member records.
        output_jsonl_path (str): Path to the output JSONL file.
    """
    data = []
    # Increase the CSV field size limit
    csv.field_size_limit(sys.maxsize)
    # Read member CSV file
    with open(member_csv_path, 'r', encoding='utf-8') as member_file:
        reader = csv.DictReader(member_file)
        for row in reader:
            data.append({'input': row['text'], 'label': 1})

    # Read non-member CSV file
    with open(non_member_csv_path, 'r', encoding='utf-8') as non_member_file:
        reader = csv.DictReader(non_member_file)
        for row in reader:
            data.append({'input': row['text'], 'label': 0})

    # Write to JSONL file
    with open(output_jsonl_path, 'w', encoding='utf-8') as jsonl_file:
        for entry in data:
            jsonl_file.write(json.dumps(entry) + '\n')


def jsonl_to_member_csv(jsonl_path: str) -> str:
    """
    Extracts labeled member samples (label == 1) from a JSONL file into a CSV.

    Args:
        jsonl_path (str): Path to the input JSONL file.

    Returns:
        str: Path to the output CSV file with only member entries.
    """
    output_dir = os.path.dirname(jsonl_path)
    base_name = os.path.splitext(os.path.basename(jsonl_path))[0]
    output_csv_path = os.path.join(output_dir, f"{base_name}_member.csv")

    with open(jsonl_path, "r", encoding="utf-8") as fin, \
            open(output_csv_path, "w", encoding="utf-8", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["text"])  # header

        for line in fin:
            record = json.loads(line)
            if int(record.get("label", 0)) == 1:
                writer.writerow([record.get("input", "")])

    print(f"Saved filtered CSV to: {output_csv_path}")
    return output_csv_path


def load_data(data_mode="WTQ", split="train", file_path=None, top_k=None):
    """
    Loads data from Hugging Face datasets and prepares member/non-member splits.

    Args:
        data_mode (str): Dataset name ("WTQ", etc.).
        split (str): Dataset split ("train", "test").
        file_path (str, optional): Base path to save CSV and JSONL outputs.
        top_k (int, optional): Number of samples to use.

    Returns:
        Tuple[str, str, str]: Paths to member CSV, non-member CSV, and output JSONL file.
    """
    if data_mode == "WTQ":
        dataset_name = "Stanford/wikitablequestions"
        dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)

    if top_k is None or top_k > len(dataset) or top_k <= 0:
        top_k = len(dataset)
    member_count = top_k // 2
    non_member_count = top_k - member_count

    # Load member data
    for i in range(member_count):
        sample = dataset[i]
        data = convert2df(sample['table'])
        data_str = convertdf2string(data)
        question = sample['question']
        answer = sample['answers']
        result.append((data_str, question, answer))
    df_member = pd.DataFrame(result, columns = ['text', 'question', 'answer'])
    if file_path:
        file_path_member = file_path.replace(".csv", "_member.csv")
        df_member.to_csv(file_path_member, index=False)
        print(f"Member data saved to {file_path_member}")
    result.clear()

    # Load non-member data
    for i in range(member_count, top_k):
        sample = dataset[i]
        data = convert2df(sample['table'])
        data_str = convertdf2string(data)
        question = sample['question']
        answer = sample['answers']
        result.append((data_str, question, answer))
    df_non_member = pd.DataFrame(result, columns = ['text', 'question', 'answer'])
    if file_path:
        file_path_non_member = file_path.replace(".csv", "_non_member.csv")
        df_non_member.to_csv(file_path_non_member, index=False)
        print(f"Non-member data saved to {file_path_non_member}")
    result.clear()

    # create jsonl file
    if file_path:
        output_jsonl_path = file_path.replace(".csv", ".jsonl")
        csv_to_jsonl(file_path_member, file_path_non_member, output_jsonl_path)
        print(f"Data saved to {output_jsonl_path}")
    if file_path:
        return file_path_member, file_path_non_member, output_jsonl_path
    return df_member, df_non_member, None


def load_data_from_datasets(data_mode, split):
    """
    Loads a dataset split from Hugging Face depending on data mode.

    Args:
        data_mode (str): One of "WTQ", "WikiSQL", "tab_fact", "TAT-QA", or "GitTables".
        split (str): Split name to load (e.g., "train", "test").

    Returns:
        datasets.Dataset: Loaded dataset.
    """
    if data_mode == "WTQ":
        dataset_name = "Stanford/wikitablequestions"
        dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
    elif data_mode == "WikiSQL":
        dataset_name = "Salesforce/wikisql"
        dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
    elif data_mode == "tab_fact":
        dataset_name = "wenhu/tab_fact"
        dataset = load_dataset(dataset_name,"tab_fact", split=split, trust_remote_code=True)
        dataset = dataset.rename_columns({
            "table_text": "table",
            "statement": "question"
        })
    elif data_mode == "TAT-QA":
        dataset_name = "next-tat/TAT-QA"
        dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
    elif data_mode == "GitTables":
        dataset_name = "yuansui/GitTables"
        dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
    return dataset


def load_data_unique_tables(data_mode="WTQ", split="train", output_dir=None, top_k=None, seed=42, use_existing_data=False, table_encoding="line_sep", max_table_size=-1):
    """
    Loads a dataset and extracts unique tables, optionally filtered by size and encoding.

    Args:
        data_mode (str): Dataset name.
        split (str): Dataset split to use.
        output_dir (str): Output directory to save CSV and JSONL files.
        top_k (int): Number of examples to use.
        seed (int): Random seed for shuffling and selection.
        use_existing_data (bool): Whether to reuse cached CSV/JSONL files if they exist.
        table_encoding (str): Format for encoding tables ("markdown", "line_sep", etc.).
        max_table_size (int): Max character length of table strings to include (-1 for unlimited).

    Returns:
        Tuple[str, str, str]: Paths to member CSV, non-member CSV, and JSONL file.
    """
    random.seed(seed)
    folder = f"{output_dir}/Datasets/{data_mode}/"
    os.makedirs(folder, exist_ok=True)
    file_path = folder + f"{data_mode}_{split}_format_{table_encoding}_k_{top_k}_unique_seed_{str(seed)}_max_table_{str(max_table_size)}.csv"
    file_path_member = file_path.replace(".csv", "_member.csv")
    file_path_non_member = file_path.replace(".csv", "_non_member.csv")
    output_jsonl_path = file_path.replace(".csv", ".jsonl")

    if use_existing_data and os.path.exists(file_path_member) and os.path.exists(file_path_non_member) and os.path.exists(output_jsonl_path):
        print(f"Using existing data from {file_path}")
        return file_path_member, file_path_non_member, output_jsonl_path

    dataset = load_data_from_datasets(data_mode, split)

    if top_k is None or top_k > len(dataset) or top_k <= 0:
        top_k = len(dataset)

    unique_tables = {}
    skiped_tables = set()
    num_unique_tables = 0
    for i in range(top_k):
        table = dataset[i]['table']
        table_df = convert_to_df(table, data_mode)
        table_str = convert_df_to_string(table_df)
        table_hash = hash(table_str)  # Use hash to track unique tables
        if max_table_size > 0 and len(table_str) > max_table_size: # Skip tables larger than max_table_size
            if table_hash not in skiped_tables:
                skiped_tables.add(table_hash)
                num_unique_tables += 1
            continue

        if table_hash not in unique_tables:
            # unique_tables[table_hash] = (table_str, dataset[i]['question'], dataset[i]['answers'])
            unique_tables[table_hash] = (table_str, dataset[i]['question'])
            num_unique_tables += 1

    all_keys = list(unique_tables.keys())
    random.shuffle(all_keys)  # Shuffle to randomly distribute tables between member and non-member
    midpoint = len(all_keys) // 2

    # Splitting into member and non-member datasets
    member_keys = all_keys[:midpoint]
    non_member_keys = all_keys[midpoint:]

    df_member = pd.DataFrame([unique_tables[key] for key in member_keys], columns=['text', 'question'])
    df_non_member = pd.DataFrame([unique_tables[key] for key in non_member_keys], columns=['text', 'question'])

    if table_encoding != "line-sep": # encode tables to different format
        df_member = tables_encoder.encode_df_of_tables(df_member, table_encoding, "text")
        df_non_member = tables_encoder.encode_df_of_tables(df_non_member, table_encoding, "text")
    print(f"Loaded {len(df_member)} member samples and {len(df_non_member)} non-member samples. \nTotal: {len(df_member) + len(df_non_member)} out of {num_unique_tables} unique tables.")
    if file_path:
        df_member.to_csv(file_path_member, index=False)
        print(f"Member data saved to {file_path_member}")

        df_non_member.to_csv(file_path_non_member, index=False)
        print(f"Non-member data saved to {file_path_non_member}")
        # create jsonl file
        csv_to_jsonl(file_path_member, file_path_non_member, output_jsonl_path)
        print(f"Data saved to {output_jsonl_path}")
        return file_path_member, file_path_non_member, output_jsonl_path

    return df_member, df_non_member, None


if __name__ == '__main__':
    # data_wtq = load_data_from_datasets(data_mode="WTQ", split="train")
    # data_wiki = load_data_from_datasets(data_mode="WikiSQL", split="train")
    # data_tab_fact = load_data_from_datasets(data_mode="tab_fact", split="tab_fact")
    # data_tat_qa = load_data_from_datasets(data_mode="TAT-QA", split="train")
    # data_git_tables = load_data_from_datasets(data_mode="GitTables", split="train")
    data_tab_fact2 = load_data_unique_tables("tab_fact", "train", output_dir="/dt/shabtaia/dt-sicpa/eyal/Tabular", seed=42, use_existing_data=False, table_encoding="markdown", max_table_size=-1)