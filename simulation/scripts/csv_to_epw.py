import pandas as pd
from pathlib import Path

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path.cwd()

CSV_FILE = BASE_DIR / "data" / "processed" / "indian_metro_weather_clean.csv"
OUTPUT_FILE = (
    BASE_DIR
    / "simulation"
    / "weather"
    / "Mumbai_indian_metro.epw"
)

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(CSV_FILE)

# Use ONLY Mumbai from the existing Indian Metro dataset
df = df[df["city"].str.lower() == "mumbai"].copy()

df = df.reset_index(drop=True)

if len(df) != 8760:
    raise ValueError(
        f"Expected 8760 Mumbai records, got {len(df)}"
    )

# =========================================================
# LOCATION
# =========================================================

latitude = 19.0760
longitude = 72.8777
timezone = 5.5
elevation = 14.0

# =========================================================
# EPW HEADER
# =========================================================

headers = [

    # LOCATION
    "LOCATION,Mumbai,Maharashtra,INDIAN_METRO,Mumbai,999999,19.0760,72.8777,5.5,14.0",

    # DESIGN CONDITIONS
    # Keep a valid design-condition section.
    "DESIGN CONDITIONS,0",

    # TYPICAL / EXTREME PERIODS
    "TYPICAL/EXTREME PERIODS,0",

    # GROUND TEMPERATURES
    # Three representative depths.
    # These are assumptions for the simulation model,
    # not additional weather observations.
    "GROUND TEMPERATURES,3,"
    "0.5,1.0,1800,800,"
    "24.0,24.5,26.0,28.0,30.0,31.0,"
    "30.5,30.0,29.0,27.0,25.0,24.0,"
    "2.0,1.0,1800,800,"
    "24.0,24.3,25.0,26.5,28.0,29.0,"
    "29.0,28.5,27.5,26.0,24.5,24.0,"
    "4.0,1.0,1800,800,"
    "24.0,24.1,24.5,25.5,27.0,28.0,"
    "28.0,27.5,26.5,25.5,24.5,24.0",

    # HOLIDAYS / DAYLIGHT SAVING
    "HOLIDAYS/DAYLIGHT SAVING,No,0,0",

    # COMMENTS
    "COMMENTS 1,Generated from the Indian Metro weather dataset.",
    "COMMENTS 2,Mumbai hourly temperature humidity dew point and wind data.",
    

    # DATA PERIODS
    "DATA PERIODS,1,1,Data,Sunday,1/1/2023,12/31/2023"
]

# =========================================================
# EPW WEATHER RECORD
# =========================================================

def make_epw_row(row, index):

    # Reconstruct date from sequential hourly record.
    timestamp = pd.Timestamp("2023-01-01") + pd.Timedelta(
        hours=index
    )

    year = timestamp.year
    month = timestamp.month
    day = timestamp.day

    # EPW uses:
    # 1 = 00:00-01:00
    # 24 = 23:00-00:00
    hour = timestamp.hour + 1

    minute = 60

    dry_bulb = float(row["air_temperature"])
    rh = max(0.1, min(100.0, float(row["relative_humidity"])))
    wind_speed = max(0.0, float(row["wind_speed"]))

    # Calculate dew point from dry-bulb temperature and RH
    # so the EPW humidity fields remain psychrometrically consistent.
    a = 17.625
    b = 243.04

    gamma = (a * dry_bulb / (b + dry_bulb)) + __import__("math").log(rh / 100.0)
    dew_point = (b * gamma) / (a - gamma)

    # -----------------------------------------------------
    # Missing source variables
    # -----------------------------------------------------

    pressure = 101325

    extraterrestrial_horizontal = 0
    extraterrestrial_direct = 0
    horizontal_infrared = 0

    global_horizontal = 0
    direct_normal = 0
    diffuse_horizontal = 0

    global_illuminance = 0
    direct_illuminance = 0
    diffuse_illuminance = 0
    zenith_luminance = 0

    wind_direction = 0

    total_sky_cover = 0
    opaque_sky_cover = 0

    visibility = 9999
    ceiling_height = 77777

    present_weather_observation = 9
    present_weather_codes = 999999999

    precipitable_water = 0
    aerosol_optical_depth = 0
    snow_depth = 0
    days_since_snowfall = 99

    albedo = 0.2
    liquid_precipitation_depth = 0
    liquid_precipitation_quantity = 0

    return [
        year,
        month,
        day,
        hour,
        minute,
        9,
        dry_bulb,
        dew_point,
        rh,
        pressure,
        extraterrestrial_horizontal,
        extraterrestrial_direct,
        horizontal_infrared,
        global_horizontal,
        direct_normal,
        diffuse_horizontal,
        global_illuminance,
        direct_illuminance,
        diffuse_illuminance,
        zenith_luminance,
        wind_direction,
        wind_speed,
        total_sky_cover,
        opaque_sky_cover,
        visibility,
        ceiling_height,
        present_weather_observation,
        present_weather_codes,
        precipitable_water,
        aerosol_optical_depth,
        snow_depth,
        days_since_snowfall,
        albedo,
        liquid_precipitation_depth,
        liquid_precipitation_quantity
    ]


# =========================================================
# WRITE EPW
# =========================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    for header in headers:
        f.write(header + "\n")

    for i, row in df.iterrows():

        epw_row = make_epw_row(row, i)

        f.write(
            ",".join(str(value) for value in epw_row)
            + "\n"
        )

print("")
print("========================================")
print("EPW CREATED")
print("========================================")
print("Source:", CSV_FILE)
print("City: Mumbai")
print("Rows:", len(df))
print("Output:", OUTPUT_FILE)
print("========================================")