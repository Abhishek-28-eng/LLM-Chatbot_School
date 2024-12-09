import os
import csv

# Define the path to your CSV file
csv_file_path = "chat_history.csv"

# Ensure the CSV file has headers
if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Input", "Response"])  


def save_to_csv(input_text, response_text):
    """
    Appends a user input and chatbot response to the CSV file.
    
    Args:
        input_text (str): The user's input message.
        response_text (str): The chatbot's response message.
    """
    with open(csv_file_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([input_text, response_text])  
