DATA_PREPROCESSING = """
Please convert the given data into a CSV format. Each row of the CSV should represent a single record and be on a separate line. The CSV should have the following columns:

store_id: The ID of the store (e.g., STORE_1).
item_id: The ID of the sold item, which always starts with the store_id (e.g., STORE_1_325).
cnt: The number of sales for the item_id on the specified date (e.g., 0).
date: The date in the format YYYY-MM-DD (e.g., 2011-01-29).
wday: The number of the weekday (e.g., 1 for Monday, 2 for Tuesday, etc.).
month: The month number (e.g., 1 for January, 2 for February, etc.).
year: The year (e.g., 2011).
event_name_1: The name of the event that happened on the specified date (e.g., SuperBowl). If no event occurred, use "not_event".
event_type_1: The type of the event (e.g., Sporting). If no event occurred, use "not_event". 
CASHBACK_STORE_1: A boolean value indicating whether cashback was used (1) or not (0) for the store.
Please ensure that:

Each record is on a new line in the CSV.
The column names are exactly as specified above.
The values in each column follow the given examples and formats.
You pay close attention to the details provided for each column.
"""

DATA_EXPLANATION = """
Please analyze the provided data and explain its structure based on the file type.
1. If the data is in CSV format:
Write a numbered list of columns with explanations for each column (e.g., "1. store_id: The ID of the store (e.g., STORE_1).").

2.If the data is in JSON format:
Explain the overall structure of the JSON data (e.g., "The JSON data is an array of objects, where each object represents a record.").
Write a numbered list of key-value pairs in each JSON object with explanations (e.g., "1. store_id: The ID of the store (e.g., STORE_1).").

3. If the data is in TXT format:
Explain the structure of the text data (e.g., "Each line in the text file represents a record, with values separated by commas.").
Write a numbered list of fields in each line with explanations (e.g., "1. store_id: The ID of the store (e.g., STORE_1).").

4. If the data is in XML format:
Explain the overall structure of the XML data (e.g., "The XML data consists of a root element named 'records', with each child element representing a record.").
Write a numbered list of XML tags with explanations (e.g., "1. : The ID of the store (e.g., STORE_1).").
"""

def GRAPH_GENERATION(prompt, filepath, colums_explanation):
    return f"""
File path: {filepath}
Columns explanation: {colums_explanation}
Please, write single python function with all the necessary imports to generate graph using matplotlib with the following information: {prompt}.
Set file path {filepath} as a default value to your graph function input and use data from it by reading - do not define any other data.
Your answer should only contain python code, without any comments.
Call the function you wrote at the end of the script.
"""
