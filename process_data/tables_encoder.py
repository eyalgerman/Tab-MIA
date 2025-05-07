import csv
import json
import re

import pandas as pd


def csv_to_list(csv_text):
    """Convert CSV text to a list of lists."""
    reader = csv.reader(csv_text.strip().split('\n'))
    return list(reader)

def list_to_key_value_pair(lines):
    """Convert a list of lists to a key-value pair string format."""
    if not lines or len(lines) < 2:
        return None
    headers = lines[0]
    rows = lines[1:]
    result = []
    for row in rows:
        pairs = []
        for i in range(min(len(headers), len(row))):
            pairs.append(f"{headers[i]}: {str(row[i]).strip()}")
        result.append(" | ".join(pairs))

    return "\n".join(result)


def list_to_key_is_value(lines):
    """Convert a list of lists to a key-is-value string format."""
    headers = lines[0]
    rows = lines[1:]
    output = []
    for row in rows:
        output.append(
            ", ".join(f"{headers[i]} is {str(row[i]).strip()}" for i in range(min(len(headers), len(row)))) + '.'
        )
    return "\n".join(output)


def list_to_html_table(data):
    """Convert a list of lists to an HTML table."""
    html = '<table>\n'
    for row in data:
        html += '  <tr>\n'
        for cell in row:
            html += '    <td>{}</td>\n'.format(cell.strip())
        html += '  </tr>\n'
    html += '</table>'
    return html


def list_to_json(data):
    """Convert a list of lists to a JSON string."""
    headers = [header.strip() for header in data[0]]
    json_list = []
    for row in data[1:]:
        padded_row = row + [""] * (len(headers) - len(row))  # fill missing values
        json_list.append({headers[i]: padded_row[i].strip() for i in range(len(headers))})
    return json.dumps(json_list, indent=4)


def is_numeric_column(column):
    """Heuristically checks if a column contains mostly numeric data (int/float or +/− prefixed)."""
    numeric_pattern = re.compile(r"^[+-]?\d+([.,]\d+)?$")
    numeric_count = sum(1 for cell in column if numeric_pattern.match(cell.strip()))
    return numeric_count >= len(column) // 2  # more than half

def list_to_markdown_table(data):
    if not data:
        return ""

    # Ensure all rows have the same number of columns as the longest row
    max_cols = max(len(row) for row in data)
    padded_data = [row + [''] * (max_cols - len(row)) for row in data]

    columns = list(zip(*padded_data))
    col_widths = [max(len(cell.strip()) for cell in col) for col in columns]
    col_widths = [max(w, 3) for w in col_widths]

    # Detect numeric-looking columns (excluding header row)
    import re
    def is_numeric_column(col):
        numeric_pattern = re.compile(r"^[+-]?\d+([.,]\d+)?$")
        return sum(1 for val in col if numeric_pattern.match(val.strip())) >= len(col) // 2

    is_numeric = [is_numeric_column(col[1:]) for col in columns]

    headers = [cell.strip() for cell in padded_data[0]]
    markdown = ''
    header_line = '| ' + ' | '.join(f'{headers[i]:<{col_widths[i]}}' for i in range(max_cols)) + ' |\n'
    separator_line = '| ' + ' | '.join('-' * col_widths[i] for i in range(max_cols)) + ' |\n'
    markdown += header_line + separator_line

    for row in padded_data[1:]:
        row_cells = [cell.strip() for cell in row]
        row_line = '| '
        for i in range(max_cols):
            cell = row_cells[i]
            if is_numeric[i]:
                row_line += f'{cell:>{col_widths[i]}} | '
            else:
                row_line += f'{cell:<{col_widths[i]}} | '
        markdown += row_line.rstrip() + '\n'

    return markdown


def encode_line_sep_table(table, encoder_method):
    """Encode a table in line-separated format using the specified encoder method."""
    data = csv_to_list(table)
    if encoder_method.lower() == 'json':
        return list_to_json(data)
    elif encoder_method.lower() == 'html':
        return list_to_html_table(data)
    elif encoder_method.lower() == 'markdown':
        return list_to_markdown_table(data)
    elif encoder_method.lower() == 'key-value-pair':
        return list_to_key_value_pair(data)
    elif encoder_method.lower() == 'key-is-value':
        return list_to_key_is_value(data)
    else:
        raise ValueError(f"Unknown encoder method: {encoder_method}")


def encode_df_of_tables(df, encoder_method, column_name):
    """Encode a DataFrame of tables using the specified encoder method."""
    # Apply the encoding function to the specified column
    df[column_name] = df[column_name].apply(lambda table: encode_line_sep_table(table, encoder_method))
    return df


def encode_csv_file(input_csv_path, output_csv_path, encoder_method, column_name):
    """
    Encode a CSV file containing tables using the specified encoder method.
    """
    # Read the CSV file
    df = pd.read_csv(input_csv_path)
    # Encode the tables
    df = encode_df_of_tables(df, encoder_method, column_name)
    # Save the encoded tables to a new CSV file
    df.to_csv(output_csv_path, index=False)
    print(f"Encoded tables saved to {output_csv_path}")


def encode_jsonl_file(input_jsonl_path, output_jsonl_path, encoder_method, column_name="input"):
    # Read the JSONL file
    df = pd.read_json(input_jsonl_path, lines=True)
    # Encode the tables
    df = encode_df_of_tables(df, encoder_method, column_name)
    # Save the encoded tables to a new JSONL file
    df.to_json(output_jsonl_path, orient='records', lines=True)
    print(f"Encoded tables saved to {output_jsonl_path}")



if __name__ == '__main__':
    # Example usage
    csv_text = """Name,Age, Occupation
    Alice,30,Engineer
    Bob,25
    Charlie,35,Teacher"""

    data = csv_to_list(csv_text)
    html_table = list_to_html_table(data)
    json_data = list_to_json(data)
    markdown_table = list_to_markdown_table(data)
    key_value_pair = list_to_key_value_pair(data)
    key_is_value = list_to_key_is_value(data)

    print(f"HTML Table: \n{html_table}")
    print(f"\nJSON Data: \n{json_data}")
    print(f"\nMarkdown Table: \n{markdown_table}")
    print(f"\nKey-Value Pair: \n{key_value_pair}")
    print(f"\nKey is Value: \n{key_is_value}")
