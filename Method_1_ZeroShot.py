import pandas as pd
import openai

# Set OpenAI API key
openai.api_key = "openai_api_key"

# Define your modified ESI Guidelines algorithm as a string.
# You can expand, reformat, or modify this string as needed to reflect your guidelines.
ESI_ALGORITHM = """
Modified ESI Guidelines Algorithm:
1. Evaluate the patient’s stability based on vital signs and overall presentation.
2. Consider the severity and acuity of the patient's chief complaint.
3. Account for patient age and any comorbidities if mentioned.
4. Assign an ESI level:
   - ESI 1: Immediate life-saving intervention required.
   - ESI 2: High risk situation, patient should be seen promptly.
   - ESI 3: Stable but requires many resources.
   - ESI 4: Stable and requires one resource.
   - ESI 5: Stable and requires no resources.
Use the guidelines above to determine the proper ESI level and provide a concise explanation for the scoring decision.
"""

def get_esi_score(narrative, algorithm_text):
    """
    Combine the patient narrative with the ESI algorithm and use OpenAI API to generate 
    an ESI score along with reasoning.
    
    Args:
        narrative (str): Patient narrative from the record.
        algorithm_text (str): The modified ESI Guidelines algorithm.
    
    Returns:
        str: The ESI output including the score and reasoning.
    """
    prompt = (
        f"Below is a modified Emergency Severity Index (ESI) Guidelines algorithm:\n"
        f"{algorithm_text}\n\n"
        f"And the patient narrative:\n{narrative}\n\n"
        "Based on the guidelines and the narrative provided, determine the appropriate ESI level "
        "for this patient (e.g., ESI 1, ESI 2, ESI 3, etc.) and include the reasoning behind the decision."
    )
    
    try:
        response = openai.Completion.create(
            engine="o1",
            prompt=prompt,
        )
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error generating ESI score: {e}")
        return "Error generating ESI output."

def process_esi_for_database(input_file, output_file):
    """
    Load the database with patient narratives from an Excel file, generate an ESI score
    and corresponding reasoning for each record, then save the new data to another Excel file.
    
    Args:
        input_file (str): File path for the existing Excel database (with a Narrative column).
        output_file (str): File path to save the new database with ESI outputs.
    """
    try:
        # Load the Excel file into a DataFrame
        df = pd.read_excel(input_file)
        print(f"Loaded {len(df)} records from {input_file}.")
    except Exception as e:
        print(f"Error reading the input file: {e}")
        return
    
    # Check that there is a 'Narrative' column
    if "Narrative" not in df.columns:
        print("The input file does not contain a 'Narrative' column.")
        return
    
    # Generate the ESI output for each record
    print("Generating ESI outputs for each record...")
    df["ESI_Output"] = df["Narrative"].apply(lambda narrative: get_esi_score(narrative, ESI_ALGORITHM))
    
    try:
        # Save the updated DataFrame to the specified output file
        df.to_excel(output_file, index=False)
        print(f"Updated database with ESI outputs saved to {output_file}.")
    except Exception as e:
        print(f"Error writing the output file: {e}")

if __name__ == "__main__":
    # Specify the file paths for input and output
    input_excel = "patient_data_with_narratives.xlsx"  # Input file should have the Narrative column.
    output_excel = "patient_data_with_ESI.xlsx"
    
    process_esi_for_database(input_excel, output_excel)