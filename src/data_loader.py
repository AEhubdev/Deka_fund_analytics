import pandas as pd
import numpy as np


def load_and_preprocess_data(file_source, date_col='Date', price_col='Price'):
    """
    Loads Excel file and prepares clean returns time series.
    Supports file paths or Streamlit UploadedFile objects.
    """
    df = pd.read_excel(file_source)

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]

    # Identify date column if alternative naming exists
    if date_col not in df.columns:
        date_candidates = [c for c in df.columns if 'date' in c.lower() or 'datum' in c.lower()]
        if date_candidates:
            date_col = date_candidates[0]
        else:
            raise KeyError(f"Date column '{date_col}' not found in Excel file.")

    # Identify price column if alternative naming exists
    if price_col not in df.columns:
        price_candidates = [c for c in df.columns if 'price' in c.lower() or 'kurs' in c.lower() or 'nav' in c.lower()]
        if price_candidates:
            price_col = price_candidates[0]
        else:
            raise KeyError(f"Price column '{price_col}' not found in Excel file.")

    # Convert types and sort chronologically
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True)
    df = df.sort_values(by=date_col).reset_index(drop=True)

    # Calculate daily returns
    df['Daily_Return'] = df[price_col].pct_change()

    return df, date_col, price_col