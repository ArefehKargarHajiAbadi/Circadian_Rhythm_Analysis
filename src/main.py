# main.py
import pandas as pd
import numpy as np
import os
import sys

# Add the project root to the system path to allow imports from 'src'
# This helps the script find the modules when run from different locations.
# Assumes main.py is in the project root or one level down if running from an IDE.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import modules from the src directory
try:
    from src.data_preparation import create_dummy_data, DataCleaner
    from src.signal_processing import SignalProcessor
    from src.cosinor_model import CosinorModel
except ImportError as e:
    print(f"Import Error: {e}. Please ensure that src/ is correctly configured and all .py files are present.")
    sys.exit(1)


# --- Configuration ---
# Use '../data/' path if running main.py from a subdirectory (like notebooks/)
# Use 'data/' path if running main.py from the project root (recommended)
RAW_FILE = 'data/raw_hr_temp_data.csv'
CLEANED_FILE = 'data/cleaned_data.csv'
FILTERED_FILE = 'data/filtered_data.csv'
VALUE_COLUMN = 'HeartRate' 


def run_pipeline():
    """Executes the entire Circadian Rhythm Analysis pipeline."""

    # --- Day 1 & 2: Data Generation and Cleaning ---
    print("\n-------------------------------------------")
    print("--- Data Generation & Cleaning ---")
    
    # Generate the raw data (Ground Truth: M=75, A=5, Acrophase=12h, Tau=24.15h)
    create_dummy_data(RAW_FILE, M_true=75, A_true=5, tau=24.15, acrophase_hour=12)

    # Clean the raw data
    cleaner = DataCleaner(RAW_FILE, CLEANED_FILE, value_column=VALUE_COLUMN)
    try:
        cleaner.load_data() \
               .handle_missing_values() \
               .handle_outliers_iqr() \
               .save_cleaned_data()
    except FileNotFoundError:
        print("Error: Raw data not found. Check file paths.")
        return

    # Load the cleaned data to define cleaned_df for the next step
    cleaned_df = pd.read_csv(CLEANED_FILE)
    cleaned_df['Timestamp'] = pd.to_datetime(cleaned_df['Timestamp'])
    cleaned_df = cleaned_df.set_index('Timestamp')
    print(" Data cleaning complete and loaded into memory.")
    
    # ---  Signal Processing and FFT ---
    print("\n-------------------------------------------")
    print("--- Filtering & FFT Analysis ---")

    processor = SignalProcessor(tau_expected=24.15) 
    
    # Day 3: Apply Butterworth Filtering
    filtered_df = processor.apply_butterworth_filter(cleaned_df, VALUE_COLUMN)
    filtered_df.to_csv(FILTERED_FILE, index=True)

    # Day 4: Perform FFT to find Tau (This defines TAU_FFT)
    TAU_FFT = processor.perform_fft_analysis(filtered_df, 'Filtered_Value')
    
    print(f" FFT Complete. Estimated Tau: {TAU_FFT:.4f} hours.")


    # --- Day 5: Cosinor Modeling ---
    print("\n-------------------------------------------")
    print("---  Cosinor Modeling ---")

    # Use the calculated TAU_FFT as the fixed period for the model
    cosinor = CosinorModel(tau=TAU_FFT)

    # Fit the model to the filtered data
    cosinor.fit_model(filtered_df, 'Filtered_Value')

    # Get the final parameters and display results
    final_results = cosinor.get_results()

    if final_results:
        print("\n--- Final Circadian Parameters ---")
        print(" (Target Truth: M=75, A=5, Acrophase=12h, Tau=24.15h) ")
        print("-" * 40)
        
        # --- Critical Correction Logic ---
        A_raw = final_results['Amplitude (A)']
        Acrophase_raw = final_results['Acrophase (Hours)']
        
        if np.sign(A_raw) < 0:
            # If Amplitude is negative, the model found the trough. Shift by Tau/2.
            Acrophase_corrected = (Acrophase_raw + (final_results['Tau (Fixed Period)'] / 2.0)) % final_results['Tau (Fixed Period)']
            A_final = np.abs(A_raw)
            correction_note = " (Corrected due to negative A)"
        else:
            Acrophase_corrected = Acrophase_raw
            A_final = A_raw
            correction_note = ""

        # Display the results with correction applied to Amplitude and Acrophase
        for key, value in final_results.items():
            if key == 'Amplitude (A)':
                print(f"| {key.ljust(20)}: **{A_final:.2f}**{correction_note}")
            elif key == 'Acrophase (Hours)':
                print(f"| {key.ljust(20)}: **{Acrophase_corrected:.2f}**{correction_note}")
            elif 'SE' in key or 'Tau' in key:
                print(f"| {key.ljust(20)}: {value:.4f}")
            elif key == 'Mesour (M)':
                 print(f"| {key.ljust(20)}: **{value:.2f}**")
        
        print("-" * 40)
        print(" Pipeline execution complete. Ready for final documentation.")


if __name__ == '__main__':
    run_pipeline()