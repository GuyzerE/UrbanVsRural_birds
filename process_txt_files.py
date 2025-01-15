import os
import pandas as pd

def process_txt_files(directory, column_name, allowed_names, replacement_name):
    """
    Reads all .txt files in a directory, processes a specified column, and replaces 
    any value not in the allowed_names list with a replacement name.

    Parameters:
        directory (str): Path to the directory containing .txt files.
        column_name (str): The column to process.
        allowed_names (list): A list of allowed string names.
        replacement_name (str): The name to replace disallowed values with.

    Returns:
        None
    """
    # Check if the directory exists
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    # List all .txt files in the directory
    txt_files = [f for f in os.listdir(directory) if f.endswith('.txt')]

    if not txt_files:
        print(f"No .txt files found in '{directory}'.")
        return

    for file in txt_files:
        file_path = os.path.join(directory, file)

        try:
            # Read the .txt file into a DataFrame
            df = pd.read_csv(file_path, delimiter='\t')

            if column_name not in df.columns:
                print(f"Column '{column_name}' not found in '{file}'. Skipping file.")
                continue

            # Replace values not in allowed_names with the replacement_name
            df[column_name] = df[column_name].apply(
                lambda x: x if x in allowed_names else replacement_name
            )

            # Save the processed DataFrame back to the file
            df.to_csv(file_path, sep='\t', index=False)
            print(f"Processed and saved file: {file}")

        except Exception as e:
            print(f"Error processing file '{file}': {e}")

# Example usage
# directory = 'path_to_directory'
# column_name = 'ColumnName'
# allowed_names = ['Name1', 'Name2', 'Name3']
# replacement_name = 'Other'
# process_txt_files(directory, column_name, allowed_names, replacement_name)


if __name__ == "__main__":
    import argparse

    # Create argument parser
    parser = argparse.ArgumentParser(description="Process .txt files in a directory.")
    parser.add_argument("directory", type=str, help="Path to the directory containing .txt files.")
    parser.add_argument("column_name", type=str, help="The column to process.")
    parser.add_argument("allowed_names", type=str, nargs='+', help="List of allowed names.")
    parser.add_argument("replacement_name", type=str, help="Name to replace disallowed values with.")

    # Parse arguments
    args = parser.parse_args()

    # Call the function with parsed arguments
    process_txt_files(args.directory, args.column_name, args.allowed_names, args.replacement_name)
