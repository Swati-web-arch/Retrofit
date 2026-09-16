import openstudio
from pathlib import Path

BASE_DIR = Path.cwd()

MODEL_FILE = BASE_DIR / "simulation" / "RetrofitIQ_mumbai.osm"
OUTPUT_FILE = BASE_DIR / "simulation" / "RetrofitIQ_ready.osm"

print("Loading model...")

model = openstudio.model.Model.load(str(MODEL_FILE)).get()

print("Model loaded successfully.")

# Get the existing RunPeriod object
run_period = model.getRunPeriod()

# Set full-year simulation
run_period.setBeginMonth(1)
run_period.setBeginDayOfMonth(1)
run_period.setEndMonth(12)
run_period.setEndDayOfMonth(31)

# Let the weather file determine the calendar
run_period.resetBeginYear()
run_period.resetEndYear()
run_period.resetDayOfWeek()

print("Run period: January 1 - December 31")

# Get the existing SimulationControl object
simulation_control = model.getSimulationControl()

# Run the weather-file simulation
simulation_control.setRunSimulationforWeatherFileRunPeriods(True)

# Do not run sizing periods yet
simulation_control.setRunSimulationforSizingPeriods(False)

print("Annual weather-file simulation enabled.")

# Save
model.save(str(OUTPUT_FILE), True)

print("Saved:", OUTPUT_FILE)