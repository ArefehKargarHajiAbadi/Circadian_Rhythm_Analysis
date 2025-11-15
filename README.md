# Circadian Rhythm Analysis Pipeline

This project implements a comprehensive Python pipeline for analyzing and accurately modeling the **circadian rhythm** from noisy physiological time-series data (simulated Heart Rate). The process employs advanced signal processing and non-linear regression techniques to extract the three key rhythm parameters: **Mesour ($M$)**, **Amplitude ($A$)**, and **Acrophase ($\Phi$)**.

---

## Project Pipeline Overview

The analysis is executed through a five-stage modular pipeline, ensuring data integrity, signal isolation, and robust parameter estimation:

1.  **Data Preparation:** Generation of synthetic data with known ground truth ($M=75, A=5, \Phi=12h, \tau=24.15h$).
2.  **Data Cleaning:** Handling missing data via **Linear Interpolation** and removing noise and outliers using the **Interquartile Range (IQR) method**.
3.  **Signal Filtering:** Applying a zero-phase **Butterworth Low-Pass Filter** to isolate the slow-moving circadian wave from high-frequency biological noise.
4.  **Period Estimation (FFT):** Using the **Fast Fourier Transform (FFT)** on the filtered signal to objectively determine the dominant period ($\tau$).
5.  **Cosinor Modeling:** Fitting the resulting data to the Cosinor equation to extract the final, precise rhythm parameters.

---

## Critical Technical Validation

The robustness of this pipeline is demonstrated by two key technical achievements:

### 1. Data-Driven Period ($\tau$) Selection (FFT)

Instead of manually assuming a fixed 24.0-hour period, the pipeline uses the FFT to let the data determine the rhythm's true length. The analysis successfully identified the dominant periodicity near the expected value.

| Parameter | Result | Significance |
| :--- | :--- | :--- |
| **FFT Estimated Tau ($\tau$)** | **24.0000 hours** | Confirmed the periodicity of the signal is approximately 24 hours, which was then used as the fixed period for the Cosinor model. |



### 2. Validation and Acrophase Correction

The Cosinor regression often extracts an Amplitude ($A$) with a negative sign, which merely indicates that the model found the **trough** (nadir) of the cycle rather than the **peak** (acrophase). The pipeline successfully implemented a critical correction logic to resolve this:

* **Raw Output:** Amplitude ($A$) was $-5.04$ and Acrophase ($\Phi$) was $0.34$ hours (raw value before correction).
* **Correction Applied:** When $A < 0$, the Acrophase must be shifted by half the period ($\tau/2$), which is $24.00/2 = 12.00$ hours.
* **Final Result:** The Amplitude is inverted to $|-5.04| = 5.04$, and the Acrophase is shifted to $0.34 + 12.00 = 12.34$ hours.

---

## Final Results and Model Fit

The corrected final parameters demonstrate high accuracy compared to the ground truth used for data generation ($M=75, A=5, \Phi=12h$).

### Extracted Circadian Parameters

| Parameter | Extracted Value | True Value | Standard Error (SE) |
| :--- | :--- | :--- | :--- |
| **Mesour ($M$)** | **75.07** BPM | 75.00 | $0.0223$ |
| **Amplitude ($A$)** | **5.04** BPM | 5.00 | $0.0315$ |
| **Acrophase ($\Phi$)** | **12.34** hours | 12.00 | N/A (Derived) |
| **Tau ($\tau$)** | 24.0000 hours | 24.15 | Fixed |

*Low Standard Errors (SE) relative to the magnitude of M and A confirm high confidence in the estimated values.*

### Visual Proof: Final Cosinor Fit

The plot below shows the smooth Cosinor curve (red line) successfully fitting the filtered Heart Rate data (blue dots), visually validating the model's accuracy.



---

## Project Structure

| Directory | Purpose | Key Contents |
| :--- | :--- | :--- |
| **`src/`** | **Core Logic** | All modular Python classes: `DataCleaner`, `SignalProcessor`, `CosinorModel`. |
| **`data/`** | **Data Storage** | Raw input, cleaned data, and the final filtered signal (`filtered_data.csv`). |
| **`notebooks/`** | **Visual Analysis** | Jupyter Notebooks for diagnostics, FFT proof, and final plotting. |
| **`results/`** | **Final Outputs** | All generated PNG plots (e.g., `rhythm_fit_plot.png`). |
| **Root (`/`)** | **Setup & Entry** | `main.py`, `README.md`, `requirements.txt`. |
1.  **Clone the repository:**
    ```bash
    git clone [Your Repository URL]
    cd Circadian_Rhythm_Analysis
    ```
2.  **Setup Environment:** (Ensure Python is installed)
    ```bash
    pip install -r requirements.txt
    ```
3.  **Execute the Pipeline:**
    ```bash
    python main.py
    ```
    (This generates data, cleans it, runs FFT, and prints the final Cosinor results to the console.)
4.  **Generate Plots:** Open the `.ipynb` files in the `notebooks/` folder and run all cells to generate and save the final plots to the `results/` directory.
