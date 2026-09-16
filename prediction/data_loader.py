import pandas as pd

from .config import DATA_PATH, TIMESTAMP_COLUMN


def load_data(path=DATA_PATH):
    """
    Load the cleaned building telemetry dataset.
    """

    df = pd.read_csv(path)

    if TIMESTAMP_COLUMN in df.columns:
        df[TIMESTAMP_COLUMN] = pd.to_datetime(
            df[TIMESTAMP_COLUMN],
            errors="coerce"
        )

        df = df.sort_values(TIMESTAMP_COLUMN)

    df = df.reset_index(drop=True)

    return df


def basic_cleaning(df):
    """
    Basic cleaning for prediction.

    The dataset is already cleaned, so this function
    performs only prediction-time safety checks.
    """

    df = df.copy()

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Remove duplicate timestamps if timestamp exists
    if TIMESTAMP_COLUMN in df.columns:
        df = df.drop_duplicates(
            subset=[TIMESTAMP_COLUMN]
        )

    return df.reset_index(drop=True)


def prepare_data(path=DATA_PATH):
    """
    Load and prepare telemetry data.
    """

    df = load_data(path)

    df = basic_cleaning(df)

    return df