#%% Import libraries
import os
import pandas as pd
import bioacoustics_model_zoo as bmz

#%% Function to extract embeddings
def embeding_extraction():
    """
    This function reads the WAV files (only the ones in the csv), extracts their embeddings, and saves them as a CSV file.
    """
    # Load HawkEars model
    model = bmz.HawkEars()

    # Define input directory and read Excel file
    input_dir = "/mnt/d/Katarina_annotations/big/" 
    file = pd.read_excel("/mnt/c/Users/guyze/OneDrive/Documents/CV4ecology/urbanization_birds/Bird identification_Golan_forGuy.xlsx",
                         sheet_name='10big') 
    
    # Prepare filenames
    file_names = (file['File name'].str.lower() + '.wav')  # Ensure lowercase extension
    filepaths = file_names.apply(lambda x: os.path.join(input_dir, x))  # Combine directory and filename


    # Check if files exist
    valid_files = [f for f in filepaths if os.path.exists(f)]
    missing_files = [f for f in filepaths if not os.path.exists(f)]
    if missing_files:
        print(f"Warning: Missing files - {missing_files}")

    # Extract embeddings in one batch
    try:
        embeddings = model.embed(valid_files)  # Process all valid files at once
        embeddings.to_csv('embeddings.csv', index=False)  # Save to CSV
        print("Embeddings saved to 'embeddings.csv'")
    except Exception as e:
        print(f"Error during embedding extraction: {e}")

#%% Main entry point
if __name__ == "__main__":
    embeding_extraction()

