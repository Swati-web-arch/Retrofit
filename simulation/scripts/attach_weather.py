import openstudio
from pathlib import Path

BASE_DIR = Path.cwd()

MODEL_FILE = BASE_DIR / "simulation" / "RetrofitIQ_baseline.osm"
WEATHER_FILE = BASE_DIR / "simulation" / "weather" / "Mumbai_indian_metro.epw"
OUTPUT_FILE = BASE_DIR / "simulation" / "RetrofitIQ_mumbai.osm"

print("Loading model...")

model = openstudio.model.Model.load(str(MODEL_FILE)).get()

print("Model loaded successfully.")
print("Weather file:", WEATHER_FILE)

# Read the EPW file
epw = openstudio.EpwFile(str(WEATHER_FILE))

print("EPW loaded successfully.")

# Attach EPW to the OpenStudio model
weather = openstudio.model.WeatherFile.setWeatherFile(model, epw)

print("Weather attached successfully.")

# Set calendar year
year_description = model.getYearDescription()
year_description.setCalendarYear(2023)

# Save the updated model
model.save(str(OUTPUT_FILE), True)

print("Saved:", OUTPUT_FILE)