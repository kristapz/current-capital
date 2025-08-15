import json
import re


def clean_text(text):
    """
    Function to clean text by removing unwanted characters and formatting issues.
    """
    # Remove newline characters and excessive spaces
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space

    # Remove non-printable characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    # Remove special characters and unicode escape sequences
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Remove sequences like '\u2212', '\u2018', etc.
    text = re.sub(r'\\u[0-9A-Fa-f]{4}', '', text)

    # Remove other unwanted special characters (if needed, customize this as needed)
    text = re.sub(r'[^\w\s.,;:?!-]', '', text)

    # Strip leading and trailing whitespace
    text = text.strip()

    return text


def clean_json_file(input_file, output_file):
    """
    Function to clean the text in a JSON file.
    """
    # Load the JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Clean the text in each topic and insight
    cleaned_data = {}
    for topic, insights in data.items():
        cleaned_insights = [clean_text(insight) for insight in insights]
        cleaned_data[topic] = cleaned_insights

    # Save the cleaned data to a new JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=4)

    print(f'Cleaned data saved to {output_file}')


# File paths for input and output JSON files
input_file = 'categorized_paragraphs.json'  # Replace with your input file path
output_file = 'cleaned_paragraphs.json'  # Replace with your desired output file path

# Clean the JSON file
clean_json_file(input_file, output_file)
