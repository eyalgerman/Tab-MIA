import json
import os
import random

import pandas as pd

from process_data import tables_encoder
from process_data.data_preparation import csv_to_jsonl
# import kaggle
def init_kaggle_package():

    # Create the .kaggle directory in the user's home directory
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)

    # Replace with your Kaggle username and API key
    api_token = {
        "username": "",  # Replace with your Kaggle username
        "key": ""  # Replace with your actual Kaggle API key
    }

    # Create the kaggle.json file with the API token
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    with open(kaggle_json_path, 'w') as file:
        json.dump(api_token, file)

    # Set the appropriate permissions for the kaggle.json file
    os.chmod(kaggle_json_path, 0o600)
    print("Kaggle API configuration complete.")


def download_kaggle_dataset(dataset_name, output_dir):
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    if not os.path.exists(kaggle_json_path):
        init_kaggle_package()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # os.system("conda activate llm")
    # os.system("export LD_LIBRARY_PATH=/home/germane/.conda/envs/llm/lib:$LD_LIBRARY_PATH")
    # Download the dataset using the Kaggle API
    download_cmd = f"kaggle datasets download -d {dataset_name} -p {output_dir} --unzip"
    os.system(download_cmd)
    print(f"Dataset '{dataset_name}' downloaded to '{output_dir}'.")

    # Unzip the downloaded dataset file
    zip_file = os.path.join(output_dir, f".zip")
    unzip_cmd = f"unzip -o {zip_file} -d {output_dir}"
    os.system(unzip_cmd)
    print(f"Dataset '{dataset_name}' unzipped.")


def chunk_csv_to_text(
    csv_path,
    chunk_size=1000,
    seed=42,
    output_dir=None,
    split=True,
    table_encoding="key-value-pair",
    use_existing_data=False,
    drop_columns=None,
):
    """
    Reads a CSV file in chunks of 'chunk_size' rows,
    converts each chunk to a text representation,
    optionally splits the chunks into two separate lists using random split,
    and saves each list as separate CSV files.

    Parameters:
    -----------
    csv_path : str
        The file path to the CSV file.
    chunk_size : int, optional
        Number of rows per chunk. Default is 1000.
    split : bool, optional
        Whether to split the text chunks into two separate lists. Default is True.
    seed : int, optional
        Random seed for splitting the data. Default is 42.
    output_dir : str, optional
        Directory to save the output CSV files. Default is 'output_chunks'.
    table_encoding : str, optional
        Encoding format for table text. Default is 'line-sep'.
    tables_encoder : object, optional
        An encoder object with a method 'encode_df_of_tables' to encode table text.
    drop_columns : list[int], optional
        Column indexes (0-based) to remove from the CSV before processing.

    Returns:
    --------
    Tuple[List[str], List[str]] or List[str]
        If split is True, returns two lists of text chunks.
        Otherwise, returns a single list of text chunks.
    """
    random.seed(seed)
    file_path_member, file_path_non_member, output_jsonl_path = None, None, None
    if output_dir is not None:
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        # Save the lists as CSV files
        data_name = os.path.basename(csv_path).split(".")[0]
        folder = f"{output_dir}/Datasets/{data_name}/"
        os.makedirs(folder, exist_ok=True)
        drop_suffix = ""
        if drop_columns:
            drop_suffix = "_drop_" + "_".join(str(i) for i in drop_columns)
        file_path = folder + f"{data_name}{drop_suffix}_format_{table_encoding}_seed_{str(seed)}_chunksize_{str(chunk_size)}.csv"
        file_path_member = os.path.join(output_dir, file_path.replace(".csv", "_member.csv"))
        file_path_non_member = os.path.join(output_dir, file_path.replace(".csv", "_non_member.csv"))
        output_jsonl_path = os.path.join(output_dir, file_path.replace(".csv", ".jsonl"))
    if use_existing_data and os.path.exists(file_path_member) and os.path.exists(file_path_non_member) and os.path.exists(output_jsonl_path):
        print(f"Using existing data from {file_path}")
        return file_path_member, file_path_non_member, output_jsonl_path
    text_chunks = []
    # Read CSV in iterable chunks
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        if drop_columns:
            cols_to_drop = [chunk.columns[i] for i in drop_columns if i < len(chunk.columns)]
            chunk = chunk.drop(columns=cols_to_drop)
        rows_as_text = []

        for _, row in chunk.iterrows():
            row_text = " | ".join(f"{col}: {row[col]}" for col in chunk.columns)
            rows_as_text.append(row_text)

        chunk_text = "\n".join(rows_as_text)
        text_chunks.append(chunk_text)
    if split:
        # Split into two separate lists randomly
        random.shuffle(text_chunks)
        mid = len(text_chunks) // 2
        chunk_list_1, chunk_list_2 = text_chunks[:mid], text_chunks[mid:]
    else:
        chunk_list_1, chunk_list_2 = text_chunks, []

    # Convert to DataFrame
    df_member = pd.DataFrame({"text": chunk_list_1})
    df_non_member = pd.DataFrame({"text": chunk_list_2})

    # Encode tables if necessary
    if table_encoding != "key-value-pair":
        df_member["text"] = df_member["text"].apply(lambda table: convert_chunk_to_line_sep(table))
        df_non_member["text"] = df_non_member["text"].apply(lambda table: convert_chunk_to_line_sep(table))
        if table_encoding != "line-sep":
            df_member = tables_encoder.encode_df_of_tables(df_member, table_encoding, "text")
            df_non_member = tables_encoder.encode_df_of_tables(df_non_member, table_encoding, "text")

    if output_dir is not None:
        df_member.to_csv(file_path_member, index=False)
        df_non_member.to_csv(file_path_non_member, index=False)

        csv_to_jsonl(file_path_member, file_path_non_member, output_jsonl_path)
        print(f"Data saved to {output_jsonl_path}")

        if not split:
            return df_member, None, output_jsonl_path

        return file_path_member, file_path_non_member, output_jsonl_path

    return df_member, df_non_member, None


def chunk_csv_with_synthetic_data(
    csv_path,
    csv_syn_path,
    chunk_size=1000,
    seed=42,
    output_dir=None,
    table_encoding="",
    use_existing_data=False,
    drop_columns=None,
):
    """
    Reads a CSV file in chunks of 'chunk_size' rows.

    Parameters are similar to ``chunk_csv_to_text`` with ``csv_syn_path`` as the
    synthetic CSV and an optional ``drop_columns`` list of column indexes to
    remove from both datasets before processing.
    """
    random.seed(seed)
    random.seed(seed)
    file_path_member, file_path_non_member, output_jsonl_path = None, None, None
    if output_dir is not None:
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        # Save the lists as CSV files
        data_name = os.path.basename(csv_path).split(".")[0]
        folder = f"{output_dir}/Datasets/{data_name}/"
        os.makedirs(folder, exist_ok=True)
        drop_suffix = ""
        if drop_columns:
            drop_suffix = "_drop_" + "_".join(str(i) for i in drop_columns)
        file_path = folder + f"{data_name}{drop_suffix}_format_{table_encoding}_seed_{str(seed)}_chunksize_{str(chunk_size)}_all_data.csv"
        file_path_member = os.path.join(output_dir, file_path.replace(".csv", "_member.csv"))
        file_path_non_member = os.path.join(output_dir, file_path.replace(".csv", "_non_member_syn.csv"))
        output_jsonl_path = os.path.join(output_dir, file_path.replace(".csv", "_with_synthetic.jsonl"))
    if use_existing_data and os.path.exists(file_path_member) and os.path.exists(
            file_path_non_member) and os.path.exists(output_jsonl_path):
        print(f"Using existing data from {file_path}")
        return file_path_member, file_path_non_member, output_jsonl_path
    chunk_list_1, chunk_list_2 = [] , []
    # Read CSV in iterable chunks
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        if drop_columns:
            cols_to_drop = [chunk.columns[i] for i in drop_columns if i < len(chunk.columns)]
            chunk = chunk.drop(columns=cols_to_drop)
        rows_as_text = []

        for _, row in chunk.iterrows():
            row_text = " | ".join(f"{col}: {row[col]}" for col in chunk.columns)
            rows_as_text.append(row_text)

        chunk_text = "\n".join(rows_as_text)
        chunk_list_1.append(chunk_text)

    for chunk in pd.read_csv(csv_syn_path, chunksize=chunk_size):
        if drop_columns:
            cols_to_drop = [chunk.columns[i] for i in drop_columns if i < len(chunk.columns)]
            chunk = chunk.drop(columns=cols_to_drop)
        rows_as_text = []

        for _, row in chunk.iterrows():
            row_text = " | ".join(f"{col}: {row[col]}" for col in chunk.columns)
            rows_as_text.append(row_text)

        chunk_text = "\n".join(rows_as_text)
        chunk_list_2.append(chunk_text)

    # make sure the two lists are of the same length
    if len(chunk_list_1) > len(chunk_list_2):
        chunk_list_1 = chunk_list_1[:len(chunk_list_2)]
    elif len(chunk_list_2) > len(chunk_list_1):
        chunk_list_2 = chunk_list_2[:len(chunk_list_1)]

    # Convert to DataFrame
    df_member = pd.DataFrame({"text": chunk_list_1})
    df_non_member = pd.DataFrame({"text": chunk_list_2})

    # Encode tables if necessary
    if table_encoding != "key-value-pair":
        df_member["text"] = df_member["text"].apply(lambda table: convert_chunk_to_line_sep(table))
        df_non_member["text"] = df_non_member["text"].apply(lambda table: convert_chunk_to_line_sep(table))
        if table_encoding != "line-sep":
            df_member = tables_encoder.encode_df_of_tables(df_member, table_encoding, "text")
            df_non_member = tables_encoder.encode_df_of_tables(df_non_member, table_encoding, "text")

    if output_dir is not None:
        df_member.to_csv(file_path_member, index=False)
        df_non_member.to_csv(file_path_non_member, index=False)

        csv_to_jsonl(file_path_member, file_path_non_member, output_jsonl_path)
        print(f"Data saved to {output_jsonl_path}")

        return file_path_member, file_path_non_member, output_jsonl_path

    return df_member, df_non_member, None


def convert_chunk_to_line_sep(table_str):
    """
    Converts a string representation of key-value table format into a CSV-style format.

    Parameters:
    -----------
    table_str : str
        The string containing a table in key-value format.

    Returns:
    --------
    str
        The table converted to CSV-style format.
    """
    lines = table_str.strip().split("\n")

    # Extract headers from the first line
    headers = [pair.split(": ")[0] for pair in lines[0].split(" | ")]

    # Extract row values
    rows = [[pair.split(": ")[1] for pair in line.split(" | ")] for line in lines]

    # Format as CSV
    csv_lines = [",".join(headers)] + [",".join(row) for row in rows]
    return "\n".join(csv_lines)


# Example usage:
if __name__ == "__main__":
    output_dir = "../../datasets/adult"
    # download_kaggle_dataset("wenruliu/adult-income-dataset", output_dir)
    chunks_as_text, _ , _= chunk_csv_to_text(os.path.join(output_dir, "adult.csv"), chunk_size=100, table_encoding="markdown")

    # You can now do something with 'chunks_as_text',
    # e.g., save them to files, or feed them into a fine-tuning pipeline.
    for i, text_chunk in enumerate(chunks_as_text["text"][:3]):
        print(f"--- Chunk {i} ---")
        print(text_chunk[:500], "...\n")  # Print just the first 500 characters
        print("\n")
        # print("Converting to line-separated format:")
        # print(convert_chunk_to_line_sep(text_chunk))
