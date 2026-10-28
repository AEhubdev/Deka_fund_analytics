import pandas as pd
import numpy as np


def load_and_preprocess_data(file_source):
    """
    Loads and parses the specific Trainee-Aufgabe_2026.xlsx structure
    for Deka-Deutschland Nebenwerte CF.
    """
    # Read the raw excel without assuming headers
    df_raw = pd.read_excel(file_source, header=None)

    # Based on file inspection, data starts at row index 3
    # Row 2 contains column headers ['Datum', 'Fondspreise des...']
    # Rows 3 onwards contain the actual time series data
    data = df_raw.iloc[3:].copy()
    data = data.iloc[:, :2]  # Keep first two columns
    data.columns = ['Date', 'Price']

    # Drop any potential NaN rows
    data = data.dropna(subset=['Date', 'Price']).reset_index(drop=True)

    # Convert types and sort chronologically
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data['Price'] = pd.to_numeric(data['Price'], errors='coerce')

    data = data.dropna().sort_values(by='Date').reset_index(drop=True)

    # Calculate daily returns
    data['Daily_Return'] = data['Price'].pct_change()

    return data, 'Date', 'Price'