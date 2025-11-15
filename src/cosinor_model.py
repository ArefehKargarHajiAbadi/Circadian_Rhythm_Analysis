# src/cosinor_model.py
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

class CosinorModel:
    """
    Class for defining the Cosinor function and performing non-linear regression 
    to estimate Mesour (M), Amplitude (A), and Acrophase (Phi).
    
    The function used is: Y(t) = M + A * cos(omega*t + phi)
    """
    def __init__(self, tau):
        """
        Initializes the model with the period tau found by FFT.
        param tau : The dominant period in hours
        """
        self.tau = tau
        # Angular frequency omega in rad/hour
        self.omega = 2 * np.pi / self.tau 
        self.params = None
        self.covariances = None
        
    
    def cosinor_func(self, t, M, A, phi):
        """
        The core Cosinor regression function.
        

        Parameters
        ----------
        t: Time in hours since start
        M ,A , phi.
            The parameters to be fitted.

        Returns
        -------
        cosinor_func

        """
        return M + A * np.cos(self.omega * t + phi)
    
    def fit_model(self, df, value_col):
         """Performs the non-linear least squares regression using curve_fit."""
         # 1. Prepare time (t) and values (Y)
         start_time = df.index.min()
         # t must be in hours from the start of the data
         t = (df.index - start_time).total_seconds() / 3600 
         Y = df[value_col].values

         # 2. Initial Guesses (p0)
         M_guess = np.mean(Y)
         A_guess = (np.max(Y) - np.min(Y)) / 2 # Initial amplitude guess
         p0 = [M_guess, A_guess, 0.0] # Initial phase (phi) guess is 0

         # 3. Perform Regression
         try:
             self.params, self.covariances = curve_fit(
                 self.cosinor_func, 
                 t, 
                 Y, 
                 p0=p0,
                 maxfev=5000 
             )
             print("Cosinor model fitting successful.")
         except RuntimeError as e:
             print(f"Error: Optimal parameters not found. {e}")
             self.params = None
             self.covariances = None
    def get_results(self):
         """Returns the estimated parameters (M, A, Acrophase in hours)."""
         if self.params is None: return None

         M, A, phi = self.params
         perr = np.sqrt(np.diag(self.covariances))
         M_se, A_se, phi_se = perr

         # 1. Acrophase Calculation: Convert phase angle (phi, radians) to time (hours)
         # acrophase_rad: Normalized phase angle [0, 2pi]
         acrophase_rad = phi % (2 * np.pi) 
         # Acrophase (hours): Time of peak activity, normalized [0, tau]
         # Note the negative sign to convert from phase shift to time of maximum
         acrophase_hours = -(acrophase_rad / self.omega) % self.tau
         
         results = {
             'Mesour (M)': M,
             'Amplitude (A)': A,
             'Acrophase (Hours)': acrophase_hours,
             'Tau (Fixed Period)': self.tau,
             'M_SE': M_se,
             'A_SE': A_se
         }
         return results
    def get_fit_curve(self, df, n_points=500):
         """Generates estimated Y values for plotting the fitted curve."""
         if self.params is None: return None
         
         start_time = df.index.min()
         end_time = df.index.max()
         
         # Create a smoothly spaced time index for the fitted curve
         time_range = pd.to_datetime(np.linspace(start_time.value, end_time.value, n_points))
         # Convert this time range back into hours since start (t)
         t_fit = (time_range - start_time).total_seconds() / 3600
         
         # Calculate the fitted value (Y) for the new smooth time range
         Y_fit = self.cosinor_func(t_fit, *self.params)
         
         fit_df = pd.DataFrame({
             'Time': time_range,
             'Fitted_Value': Y_fit
         }).set_index('Time')
         
         return fit_df
