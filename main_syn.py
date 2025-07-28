import os

from MIA import mia_detection
from options import Options
from process_data import process_csv_file, tables_encoder
from process_data.data_preparation import load_data_unique_tables


def main_short_tables(args, table_encoding):
    """
    Runs membership inference attack (MIA) evaluation on short table formats with a specified encoding.

    If the input data is in "line-sep" format and the desired encoding is different, the function
    re-encodes the data to the target format before proceeding with MIA evaluation.

    Args:
        args (Namespace): Parsed command-line arguments with fields like:
            - data (str): Path to the input `.jsonl` file.
            - target_model (str): Path to the fine-tuned model.
            - output_dir (str): Directory to store results.
            - use_existing (str): Flag indicating whether to reuse existing data ("all", "data", etc.).
        table_encoding (str): The target table encoding format to apply (e.g., "markdown", "html").
    """
    print(f"Starting the process on encoding: {table_encoding}")
    use_existing_data = True if args.use_existing.lower() in ['all', 'data'] else False
    if "line-sep" in args.data and table_encoding != "line-sep":
        line_sep_data = args.data
        data_encoded = args.data.replace("line-sep", table_encoding)
        if use_existing_data and not os.path.exists(data_encoded):
            print(f"Encoding tables to {table_encoding} format...")
            tables_encoder.encode_jsonl_file(line_sep_data, data_encoded, table_encoding)
    else:
        data_encoded = args.data
    mia_detection.main(model_path=args.target_model, data_path=data_encoded, output_dir=args.output_dir)


def main(args):
    """
    Run MIA evaluation across multiple table encoding formats.
    Compare the original data with the synthetic data to determine if the model has memorized the training data.

    Args:
        args (Namespace): Parsed command-line arguments with attributes:
            - data (str): Path to input dataset (.csv or .jsonl).
            - syn_data (str): Path to synthetic data for use with CSV input.
            - target_model (str): Path to the fine-tuned model.
            - output_dir (str): Directory for output files and results.
            - seed (int): Random seed for reproducibility.
            - max_table_size (int): Maximum rows per table for chunking.
            - use_existing (str): Indicator for reusing existing data ("all", "data", etc.).

    Raises:
        NotImplementedError: If the input file format is not supported.
    """
    output_dir = args.output_dir
    print(f"Starting the process on mode: {args.data}")
    use_existing_data = True if args.use_existing.lower() in ['all', 'data'] else False
    encoders = ['key-is-value', 'markdown', 'json', 'html', 'key-value-pair', 'line-sep']
    # tarin_jsonl_path = output_jsonl_path
    for enc in encoders:
        ext = os.path.splitext(args.data)[1].lower()
        if ext == ".csv":
            file_path_member, file_path_non_member, output_jsonl_path = process_csv_file.chunk_csv_with_synthetic_data(
                args.data,
                args.syn_data,
                output_dir=output_dir,
                seed=args.seed,
                table_encoding=enc,
                use_existing_data=use_existing_data,
                chunk_size=args.max_table_size,
                drop_columns=args.drop_columns,
            )
            mia_detection.main(model_path=args.target_model, data_path=output_jsonl_path, output_dir=args.output_dir)
        elif ext == ".jsonl":
            main_short_tables(args, enc)
        else:
            print(f"Data format not supported: {args.data}")
            raise NotImplementedError('Only CSV and JSONL files are supported for now.')
    return None


if __name__ == '__main__':
    args = Options()
    args = args.parser.parse_args()
    if args.drop_columns:
        args.drop_columns = [int(i) for i in args.drop_columns.split(',') if i.strip()]
    else:
        args.drop_columns = []
    main(args)
