import glob
import json
import os
import re
from datetime import datetime
from datasets import load_dataset
from huggingface_hub import login
import QLora_Medium_Finetune_LLM
from MIA import mia_detection
from options import Options
from process_data import process_csv_file, data_preparation
from process_data.data_preparation import load_data_unique_tables


def extract_timestamp_from_folder(folder_name):
    """
    Extracts a timestamp from a folder name using a predefined pattern.

    The expected timestamp format in the folder name is: YYYY_MM_DD_HH_MM_SS.

    Args:
        folder_name (str): The name of the folder to extract the timestamp from.

    Returns:
        datetime or None: A datetime object if a valid timestamp is found and parsed,
        otherwise None.
    """
    # Define the regex pattern for the timestamp
    timestamp_pattern = r'\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}'

    # Search for the pattern in the folder name
    match = re.search(timestamp_pattern, folder_name)

    if match:
        timestamp_str = match.group()
        try:
            # Convert to datetime object
            return datetime.strptime(timestamp_str, '%Y_%m_%d_%H_%M_%S')
        except ValueError:
            return None
    else:
        return None


def find_existing_model_folder(model_name, data_name, num_epochs, directory):
    """
    Searches for an existing model folder matching a naming convention and returns the most recent one.

    The folder name is expected to match the pattern:
    `<model_name>_<data_name>_QLORA_<timestamp>_epochs_<num_epochs>_Merged`.

    Args:
        model_name (str): Name or path of the base model.
        data_name (str): Name or path of the dataset.
        num_epochs (int): Number of training epochs used in model training.
        directory (str): Base directory where the "Models" subdirectory is located.

    Returns:
        str or None: The path to the most recently created matching model folder, or None if no match is found.
    """
    # Construct the pattern, ignoring the current_time part
    directory += '/Models'
    model_name_base = model_name.split('/')[-1]
    data_name = data_name.split('/')[-1]
    data_name = data_name.split('.')[0]
    search_pattern = f"{model_name_base}_{data_name}_QLORA_*_epochs_{num_epochs}_Merged"
    # Create a search pattern for folders
    search_path = os.path.join(directory, search_pattern)
    # Use glob to find matching folders
    matching_folders = glob.glob(search_path)
    print(f'Found {len(matching_folders)} matching folders for {model_name_base} and {data_name}')
    print(f'Searching for: {search_path}')
    # Return the first match found, or None if no folder matches
    if matching_folders:
        # Extract timestamps and sort by the newest
        folders_with_timestamps = [(folder, extract_timestamp_from_folder(folder)) for folder in matching_folders]
        # Filter out folders where timestamp extraction failed
        valid_folders = [(folder, ts) for folder, ts in folders_with_timestamps if ts is not None]

        if valid_folders:
            # Sort by timestamp descending
            valid_folders.sort(key=lambda x: x[1], reverse=True)
            newest_folder = valid_folders[0][0]
            print(f"Found newest model folder: {newest_folder}")
            return newest_folder
        else:
            print("No valid timestamps found in the folder names.")
            return None
    else:
        return None


def get_hf_dataset(dataset_name: str, encoding: str, output_folder: str) -> (str, str):
    """
    Downloads a dataset from Hugging Face and writes it to a local `.jsonl` file.
    Also creates a CSV file from the member data for later use.

    Args:
        dataset_name (str): Name of the Hugging Face dataset to load.
        encoding (str): Format in which to encode the tables (e.g., "markdown", "html").
        output_folder (str): Path to the folder where output files should be saved.

    Returns:
        Tuple[str, str]: A tuple containing the path to the local `.jsonl` file
        and the path to the member CSV file.
    """
    login(token=os.getenv("HUGGINGFACE_HUB_TOKEN")) #hagginface token
    ds = load_dataset("germane/Tab-MIA", name=dataset_name, split=encoding)
    local_path = f"{output_folder}/{dataset_name}_format_{encoding}.jsonl"
    with open(local_path, "w", encoding="utf-8") as f:
        for record in ds:
            f.write(json.dumps(record) + "\n")
    print(f"Saved dataset to: {local_path}")
    # Create a CSV file for the member data
    member_csv_path = data_preparation.jsonl_to_member_csv(local_path)
    return local_path, member_csv_path


def main(args):
    """
    Executes the full data preparation, model fine-tuning, and evaluation pipeline.

    Depending on the input data source (CSV, Hugging Face dataset, or predefined table mode),
    this function prepares the data, fine-tunes a QLoRA model (if no existing model is found),
    and runs membership inference attack (MIA) detection. It also supports reusing existing data
    or models based on user configuration.

    Args:
        args (Namespace): Parsed command-line arguments containing the following attributes:
            - output_dir (str): Directory to save outputs such as models and processed data.
            - data (str): Dataset identifier or path (e.g., path to CSV or "tabMIA_datasetname").
            - use_existing (str): Whether to reuse existing data/model ("all", "data", "model", or "none").
            - seed (int): Random seed for reproducibility.
            - table_encoding (str): Table encoding format (e.g., "markdown", "html", etc.).
            - max_table_size (int): Maximum number of rows per table chunk.
            - split (str): Dataset split (e.g., "train", "test").
            - top_k (int): Number of top tables to use from the dataset.
            - target_model (str): Path or name of the base model to fine-tune.
            - num_epochs (int): Number of training epochs for model fine-tuning.

    Raises:
        Exception: Any unexpected error during data processing or MIA detection.
    """
    output_dir = args.output_dir
    print(f"Starting the process on mode: {args.data}")
    use_existing_data = True if args.use_existing.lower() in ['all', 'data'] else False
    if args.data[-4:] == ".csv":
        file_path_member, file_path_non_member, output_jsonl_path = process_csv_file.chunk_csv_to_text(args.data, output_dir=output_dir, seed=args.seed, table_encoding=args.table_encoding, use_existing_data=use_existing_data, chunk_size=args.max_table_size)
    elif args.data.startswith("tabMIA_"):
        # Load dataset from Hugging Face, e.g., "tabMIA_adult"
        dataset_name = args.data.split("_", 1)[1]  # safely extract after the first "_"
        encoding = args.table_encoding.replace("-", "_")
        output_jsonl_path, file_path_member = get_hf_dataset(dataset_name, encoding, output_dir + "/Datasets/")
    else:
        file_path_member, file_path_non_member, output_jsonl_path = load_data_unique_tables(data_mode=args.data, split=args.split, output_dir=output_dir, top_k=args.top_k, seed=args.seed, use_existing_data=use_existing_data, table_encoding=args.table_encoding, max_table_size=args.max_table_size)
    # fine-tune the model
    use_existing_model = True if args.use_existing.lower() in ['all', 'model'] else False
    new_model = None
    base_model = args.target_model
    if use_existing_model:
        new_model = find_existing_model_folder(args.target_model, file_path_member, args.num_epochs, directory=output_dir)
        if new_model:
            args.target_model = new_model
            print(f"Using existing model from folder: {new_model}")
        else:
            print("No existing model found. Training a new model.")
    if not new_model:
        new_model = QLora_Medium_Finetune_LLM.main(args.target_model, file_path_member, output_dir + "/Models/", args.num_epochs)
    args.target_model = new_model
    encoders = ['key-value-pair', 'line-sep', 'markdown', 'html', 'json', 'key-is-value']
    encoders = [args.table_encoding]
    tarin_jsonl_path = output_jsonl_path
    for enc in encoders:
        try:
            if enc == args.table_encoding:
                output_jsonl_path = tarin_jsonl_path
            elif args.data[-4:] == ".csv":
                file_path_member, file_path_non_member, output_jsonl_path = process_csv_file.chunk_csv_to_text(args.data, output_dir=output_dir, seed=args.seed, table_encoding=enc, use_existing_data=True, chunk_size=args.max_table_size)
            elif args.data.startswith("tabMIA_"):
                # Load dataset from Hugging Face, e.g., "tabMIA_adult"
                dataset_name = args.data.split("_", 1)[1]  # safely extract after the first "_"
                encoding = enc.replace("-", "_")
                output_jsonl_path, file_path_member = get_hf_dataset(dataset_name, encoding, output_dir + "/Datasets/")
            else:
                file_path_member, file_path_non_member, output_jsonl_path = load_data_unique_tables(data_mode=args.data, split=args.split, output_dir=output_dir, top_k=args.top_k, seed=args.seed, use_existing_data=True, table_encoding=enc, max_table_size=args.max_table_size)
            # Check if it has a metrics results file
            metrics_path = None
            if metrics_path is None and args.use_existing == 'all':
                mia_detection.main(model_path=new_model, data_path=output_jsonl_path, output_dir=args.output_dir)
            else:
                print(f"Metrics file already exists: {metrics_path}")
        except Exception as e:
            print(f"Error processing encoder {enc}: \n{e}")
            raise e


if __name__ == '__main__':
    args = Options()
    args = args.parser.parse_args()
    main(args)