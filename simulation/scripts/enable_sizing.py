import openstudio
from pathlib import Path

BASE_DIR = Path.cwd()

INPUT_FILE = BASE_DIR / "simulation" / "RetrofitIQ_baseline_hvac_fixed.osm"
OUTPUT_FILE = BASE_DIR / "simulation" / "RetrofitIQ_baseline_hvac_sized.osm"

print("=" * 50)
print("ENABLING HVAC SYSTEM SIZING")
print("=" * 50)

model = openstudio.model.Model.load(str(INPUT_FILE)).get()

print("Model loaded.")

simulation_control = model.getSimulationControl()

# Enable the actual sizing calculations
simulation_control.setDoZoneSizingCalculation(True)
simulation_control.setDoSystemSizingCalculation(True)
simulation_control.setDoPlantSizingCalculation(True)

# Also allow sizing-period simulation
simulation_control.setRunSimulationforSizingPeriods(True)
simulation_control.setRunSimulationforWeatherFileRunPeriods(True)

print("Zone sizing calculation: True")
print("System sizing calculation: True")
print("Plant sizing calculation: True")
print("Sizing-period run: True")
print("Weather-file run: True")

model.save(str(OUTPUT_FILE), True)

print()
print("=" * 50)
print("SIZING SETTINGS CREATED")
print("=" * 50)
print("Output:", OUTPUT_FILE)
print("=" * 50)