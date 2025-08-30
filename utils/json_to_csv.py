import json
import csv
import os

def json_to_csv(json_filename: str, csv_filename: str = None) -> str:
    """
    Converts a JSON file to a CSV file.

    :param json_filename: The path to the input JSON file.
    :param csv_filename: The path to the output CSV file (optional). If None, a default filename is used.
    :return: The path to the generated CSV file.
    """
    # If no CSV filename is provided, generate a default one based on the JSON filename
    if csv_filename is None:
        base_name = os.path.splitext(os.path.basename(json_filename))[0]
        csv_filename = f"{base_name}.csv"
    
    # Read JSON data from the file
    with open(json_filename, 'r') as file:
        json_data = json.load(file)

    # Open the CSV file for writing
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=json_data[0].keys())
        
        # Write the header (field names)
        writer.writeheader()
        
        # Write the data rows
        writer.writerows(json_data)

    return csv_filename