from pathlib import Path


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bldg59_master_hourly_clean.csv"
)


# ---------------------------------------------------------
# MODEL SETTINGS
# ---------------------------------------------------------

TARGET_COLUMN = "total_kw"

TIMESTAMP_COLUMN = "date"

# Number of historical rows used to train the model.
TRAINING_FRACTION = 0.80

RANDOM_STATE = 42


# ---------------------------------------------------------
# BUILDING / COST SETTINGS
# ---------------------------------------------------------

DEFAULT_TARIFF = 9.0  # INR / kWh


# ---------------------------------------------------------
# ALERT THRESHOLDS
# ---------------------------------------------------------

# Difference between actual and expected energy consumption.

WATCH_DEVIATION = 10.0       # %
WARNING_DEVIATION = 20.0     # %
CRITICAL_DEVIATION = 35.0    # %


# Number of consecutive abnormal readings required
# before escalating a trend-based alert.

MIN_ABNORMAL_READINGS = 3


# ---------------------------------------------------------
# HEALTH SCORE
# ---------------------------------------------------------

HEALTH_MAX = 100
HEALTH_MIN = 0