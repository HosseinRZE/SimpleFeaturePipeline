import json
import csv
import io

def json_to_csv_in_memory(json_filename: str) -> str:
    """
    Converts a JSON file to a CSV string in memory (no disk write).

    :param json_filename: Path to input JSON file.
    :return: CSV content as a string.
    """
    # Load JSON data
    with open(json_filename, 'r') as f:
        json_data = json.load(f)

    # Use StringIO as an in-memory file
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=json_data[0].keys())
    writer.writeheader()
    writer.writerows(json_data)

    # Get CSV string
    csv_content = output.getvalue()
    output.close()
    return csv_content
