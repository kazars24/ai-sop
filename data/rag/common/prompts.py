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

GRAPH_EXPLANATION = """
Analyze the matplotlib graph providing short summarization, focusing on S&OP (Sales and Operations Planning) information. Please provide a comprehensive description of the graph, including the following elements:

Graph type: Identify whether it's a line graph, bar chart, scatter plot, or any other type of visualization.
Title and axes: Describe the graph's title and label the X and Y axes, including units of measurement if present.
Time frame: Specify the time period covered in the graph (e.g., monthly, quarterly, yearly).
Data series: Identify and describe each data series present in the graph, including their colors and markers.
Legend: If present, describe the legend and what each item represents.
Key metrics: Identify and describe the main S&OP metrics shown
Trends and patterns: Describe any visible trends, patterns, or notable fluctuations in the data.
Correlations: Identify any apparent correlations between different data series.
Anomalies: Point out any outliers or unusual data points.
Performance indicators: Describe how actual performance compares to forecasts or targets.
Seasonality: If applicable, identify any seasonal patterns in the data.
Key events: Note any significant events or milestones marked on the graph.
Data range: Specify the minimum and maximum values for each axis.
Additional visual elements: Describe any annotations, gridlines, or other visual elements present in the graph.
Color scheme: Comment on the overall color scheme and its effectiveness in conveying information.
Data density: Assess the amount of information presented and whether it's easy to interpret.
Gaps or missing data: Identify any gaps or missing data points in the graph.
Comparison points: If multiple graphs are present, compare and contrast their information.
Implications: Based on the data presented, provide insights into potential business implications for sales and operations planning.
Suggestions for improvement: If applicable, suggest ways to enhance the graph's clarity or information presentation.
Please provide a clear and concise short summary of your observations, ensuring that all relevant S&OP information is accurately described and interpreted.
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
Very important, remember: call the function that you wrote at the end of the script.
Your answer should looke like this:
import matplotlib
def func(...): # pass the file path
    ...
func() # calling the function
"""
