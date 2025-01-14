#pip install pandas openpyxl

import pandas as pd
import openai

# Set OpenAI API key
openai.api_key = "your_openai_api_key"

#Function to convert excel file into dataframe
def load_excel_to_dataframe(file_path, sheet_name=0):
    """
    Load data from an Excel file and convert it into a pandas DataFrame.

    Args:
        file_path (str): The path to the Excel file.
        sheet_name (str or int, optional): Sheet name or index to load. Default is the first sheet.

    Returns:
        pd.DataFrame: The loaded data as a DataFrame.
    """
    try:
        dataframe = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"Data successfully loaded from {file_path}")
        return dataframe
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None

def generate_patient_narrative(row):
    """
    Generate a short narrative for a patient using OpenAI's API.

    Args:
        row (pd.Series): A row of patient data from the DataFrame.

    Returns:
        str: The generated narrative.
    """
    #These are just example variable names. The actual variable names will come from the dataset.
    age = row.get("Patient_Age", "unknown age")
    sex = row.get("Patient_Sex", "unknown sex")
    complaint = row.get("PatientChiefComplaint", "no chief complaint listed")
    mode_of_arrival = row.get("Mode of Arrival", "unknown mode of arrival")
    arrival_time = row.get("Arrival Time", "unknown time")
    vital_signs = row.get("Vital_Signs", "no vital signs available")

    prompt = (
        f"Create a short narrative for a patient presenting to the ED based on the following details:\n"
        f"Age: {age}\n"
        f"Sex: {sex}\n"
        f"Chief Complaint: {complaint}\n"
        f"Mode of Arrival: {mode_of_arrival}\n"
        f"Arrival Time: {arrival_time}\n"
        f"Vital Signs: {vital_signs}\n"
        f"The narrative should be concise and formatted like a case summary."
    )

    try:
        response = openai.Completion.create(
            engine="gpt-4o",
            prompt=prompt,
            temperature=0.7 #Can be adjusted
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error generating narrative: {e}")
        return "Error generating narrative."

if __name__ == "__main__":
    # Specify the file path to the Excel file
    excel_file = "patient_data.xlsx"  # Replace with your Excel file path
    sheet = 0  # Specify the sheet name or index

    # Load the Excel data into a DataFrame
    df = load_excel_to_dataframe(excel_file, sheet_name=sheet)

    if df is not None:
        print("Generating narratives for each patient...")
        # Generate triage narratives for each patient and add them to a new 'Narrative' column
        df["Narrative"] = df.apply(generate_patient_narrative, axis=1)

        # Save the updated DataFrame to a new Excel file
        output_file = "patient_data_with_narratives.xlsx"
        df.to_excel(output_file, index=False)
        print(f"Updated data with narratives has been saved to {output_file}")

        # Optionally, display a preview of the relevant columns
        preview_cols = ["PatientChiefComplaint", "Narrative"]
        print(df[preview_cols].head())