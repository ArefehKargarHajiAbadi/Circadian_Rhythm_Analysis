# src/signal_processing.py

from scipy import signal
import numpy as np
import pandas as pd

class SignalProcessor:
    """
    Handles signal processing steps: calculating sampling rate, 
    Butterworth filtering, and performing FFT analysis to find the dominant period (tau).
    """
    def __init__(self, tau_expected=24.15):
        """
        Initializes the processor. tau_expected is the expected rhythm period in hours
        """
        self.tau_expected = tau_expected
        self.sampling_rate = None
        self.dt_seconds = None
        self.tau_fft = None # Stores the period found by FFT
        
    def calculate_sampling_rate(self, df):
        """
        Calculates the sampling rate (Hz) and time step (seconds) 
        from the DataFrame index (assuming constant sampling).
        """
        # Ensure 'Timestamp' is the index
        if df.index.name != 'Timestamp':
           df = df.set_index('Timestamp')
        
        # Calculate the time difference between samples and find the mode
        time_diff = df.index.to_series().diff().dropna()
        
        if time_diff.empty:
            raise ValueError("Data has only one timestamp or is invalid.")
        
        self.dt_seconds = time_diff.mode().dt.total_seconds().iloc[0]
        self.sampling_rate = 1.0 / self.dt_seconds
        
        print(f"Sampling Rate (Fs): {self.sampling_rate:.4f} Hz.")
        return self
        
    def apply_butterworth_filter(self, df, value_col, order=4):
        """
         Applies a zero-phase (filtfilt) low-pass Butterworth filter to smooth the data.

         Parameters
         ----------
         df : DataFrame with the cleaned data (Timestamp index).
         value_col :The column containing the physiological data
         order : The order of the filter. The default is 4.

         Returns
         -------
         DataFrame with the added 'Filtered_Value' column.

         """
        if self.sampling_rate is None:
            self.calculate_sampling_rate(df)
            
        # Cutoff Frequency (Hz) = 1 / (T_cutoff in seconds)
        cutoff_freq_hz = 1.0 / (4 * 3600) 
        nyquist_freq = 0.5 * self.sampling_rate
        normalized_cutoff = cutoff_freq_hz / nyquist_freq
        
        # Design the Butterworth filter
        b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
        
        # Apply the filter using filtfilt (zero-phase filtering to prevent phase shift)
        filtered_values = signal.filtfilt(b, a, df[value_col].values)
        
        df_filtered = df.copy()
        df_filtered['Filtered_Value'] = filtered_values
        
        print(f"Butterworth filter (Order={order}) applied successfully. High-frequency noise reduced.")
        
        return df_filtered
    
    def perform_fft_analysis(self, df_filtered, value_col):
        """
        
        Performs Fast Fourier Transform (FFT) to identify the dominant period (tau).

        Parameters
        ----------
        df_filtered : dataFrame
            DataFrame with the Butterworth filtered values.
        value_col : dataFrame
            The column containing the filtered physiological data.

        Returns
        -------
        The dominant period (tau) in hours.

        """
        if self.sampling_rate is None:
            self.calculate_sampling_rate(df_filtered)
        
        Y = df_filtered[value_col].values
        N = len(Y) #Total number of data points
        
        # 1. FFT
        ft = np.fft.fft(Y)
        
        # 2. power spectrum
        power_spectrum = np.abs(ft[1:N//2])
        
        # 3. Calc Frequency
        frequencies = np.fft.fftfreq(N, d=self.dt_seconds)[1:N//2]
        
        # 4. Find the dominant frequency
        dominant_freq_index = np.argmax(power_spectrum)
        dominant_freq_hz = frequencies[dominant_freq_index]
        
        # 5. Convert frequency to period_Tau in hours
        tau_hours = 1.0 / (dominant_freq_hz * 3600.0)
        
        self.tau_fft = tau_hours
        print("\n--- FFT Analysis Results ---")
        print(f"Estimated Period (Tau): {tau_hours:.4f} hours")
        return tau_hours