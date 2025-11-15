import pandas as pd
import numpy as np
import os

# --- 1.Synthetic Data Generation Function ---
def create_dummy_data(file_path, M_true=75, A_true=5, tau=24.15, acrophase_hour=12):
    """
    Creates a dummy CSV file with simulated HR data, including noise, 
    outliers, and missing values.

    Parameters
    ----------
    file_path : string
        Path where the CSV file will be saved (e.g., 'data/raw_hr_temp_data.csv').
    M_true : float
        True Mesour (baseline mean) for the synthetic rhythm.. The default is 75.
    A_true : float
        True Amplitude for the synthetic rhythm. The default is 5.
    tau : float
        True Period (in hours) for the synthetic rhythm (e.g., 24.15 for FD).. The default is 24.15.
    acrophase_hour : float
        The true hour of the peak (acrophase) of the rhythm.. The default is 12.

    Returns
    -------
    A pandas DataFrame containing the generated synthetic data.

    """
    total_days = 7 # Duration of the simulated data in days
    sampling_rate_per_hour = 4 # 15-minute intervals (4 samples per hour)
    total_points = total_days * 24 * sampling_rate_per_hour
    
    # 1. Generate time points in hours (t_hours)
    t_hours = np.linspace(0, total_days*24.0 , total_points, endpoint=False, dtype=float)
    
    # 2. Calculate true phase parameter (phi_true) and generate ideal cosine wave
    omega = 2.0 * np.pi / float(tau)
    # The phase (phi) should result in a peak at 'acrophase_hour'
    phi_true = -omega * float(acrophase_hour) 
    
    Y_ideal = M_true + A_true * np.cos(omega * t_hours + phi_true)
    
    # 3. Add random noise to make it more realistic
    values = Y_ideal + np.random.normal(0.0, 1.5, total_points) # Noise with std dev 1.5
    
    # 4. Introduce artificial Outliers and Missing Data for testing cleaning process
    # Low outlier
    values[100] = 30    
    # High outlier
    values[500] = 150   
    # Block of missing data (NaNs)
    values[700:705] = np.nan 
    
    # 5. Create correct Timestamps: start_date + timedelta (Fix for TypeError)
    start_date = pd.to_datetime('2025-01-01 00:00:00')
    timestamps = start_date + pd.to_timedelta(t_hours, unit='h')
  
    # Create the DataFrame
    dummy_df = pd.DataFrame({
      'Timestamp': timestamps,
      'HeartRate': values
      })
  
    # Ensure the 'data' directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save the DataFrame to a CSV file
    dummy_df.to_csv(file_path, index=False)
    print(f"--- Dummy data created at {file_path} ---")
    print("--- True parameters used for generation: ---")
    print(f"Mesour (M): {M_true}")
    print(f"Amplitude (A): {A_true}")
    print(f"Period (Tau): {tau} hours")
    print(f"Acrophase (Peak): {acrophase_hour} hours")
    print(f"Raw data saved to {file_path}")
    
    return dummy_df # Return the DataFrame just in case you want to inspect it directly

# --- 2. Data Cleaning Class ---

class DataCleaner:
    """
    A class for cleaning physiological time series data, including 
    handling missing values and clipping outliers.
    """
    def __init__(self, raw_file_path, cleaned_file_path, value_column='HeartRate'):
        self.raw_file_path = raw_file_path
        self.cleaned_file_path = cleaned_file_path
        self.value_column = value_column
        self.df = None
        self.outliers_clipped = 0
        
        
    def load_data(self):
        """Loads data from the raw CSV file and converts Timestamp."""
        if not os.path.exists(self.raw_file_path):
            raise FileNotFoundError(f"Raw data file not found at: {self.raw_file_path}")
        
        self.df = pd.read_csv(self.raw_file_path)
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'])
        # Set Timestamp as index for cleaner time series processing later
        self.df = self.df.set_index('Timestamp')
        print("Data loaded successfully.")
        return self 
    
    def handle_missing_values(self, method='linear'):
        """Fills missing values using linear interpolation."""
        missing_count = self.df[self.value_column].isnull().sum()
        if missing_count > 0 :
            # Linear interpolation preserves the trend between available points
            self.df[self.value_column] = self.df[self.value_column].interpolation(method=method)
            print(f"Handled {missing_count} missing values using {method} interpolation.")
        else:
            print("No missing values found.")
        return self
    
    def handle_outliers_iqr(self, iqr_factor=1.5):
        """
        Clips outliers using the Interquartile Range (IQR) method.
        Outliers are replaced by the upper/lower bounds (clipping).
        """
       #  1. Calculate IQR bounds (using the current data)
        Q1 = self.df[self.value_column].quantile(0.25)
        Q3 = self.df[self.value_column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR
        
        # 2. Count and clip the outliers
        lower_outliers = (self.df[self.value_column] < lower_bound).sum()
        upper_outliers = (self.df[self.value_column] > upper_bound).sum()
        self.outliers_clipped = lower_outliers + upper_outliers
        
        # Clipping: replace extreme values with the boundary values
        self.df[self.value_column] = self.df[self.value_column].clip(lower=lower_bound, upper=upper_bound)
        
        if self.outliers_clipped > 0:
            print(f"Clipped {self.outliers_clipped} outliers (Low: {lower_outliers}, High: {upper_outliers}) using IQR factor {iqr_factor}.")
        else:
            print("No significant outliers found.")
        return self
    
    def save_cleaned_data(self):
        """Saves the cleaned DataFrame to a new CSV file."""
        os.makedirs(os.path.dirname(self.cleaned_file_path), exist_ok=True)
        self.df.to_csv(self.cleaned_file_path, index=True) # index=True saves the Timestamp index
        print(f"Cleaned data saved successfully to {self.cleaned_file_path}.")
        return self
    
# --- Example Execution (for testing the module directly) ---
if __name__ == '__main__':
    RAW_FILE = '../data/raw_hr_temp_data.csv' # Assuming execution from src/
    CLEANED_FILE = '../data/cleaned_data.csv'
    
    # 1. Generate Data
    create_dummy_data(RAW_FILE)
    
    # 2. Clean Data
    cleaner = DataCleaner(RAW_FILE, CLEANED_FILE)
    
    try:
        cleaner.load_data() \
               .handle_missing_values() \
               .handle_outliers_iqr() \
               .save_cleaned_data()
        
        print("\nPipeline check: Data preparation complete.")
        
    except FileNotFoundError as e:
        print(f"\nError during cleaning: {e}")