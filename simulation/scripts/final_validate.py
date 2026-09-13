import openstudio
from pathlib import Path

MODEL_FILE = Path.cwd() / "simulation" / "RetrofitIQ_ready.osm"

print("Loading final model...")

model = openstudio.model.Model.load(str(MODEL_FILE)).get()

print("MODEL LOAD SUCCESS")
print("Spaces:", len(model.getSpaces()))
print("Thermal Zones:", len(model.getThermalZones()))
print("Ideal Loads Systems:", len(model.getZoneHVACIdealLoadsAirSystems()))

weather = model.getOptionalWeatherFile()

if weather.is_initialized():
    wf = weather.get()
    print("Weather: ATTACHED")
    print("City:", wf.city())
    print("Latitude:", wf.latitude())
    print("Longitude:", wf.longitude())
else:
    print("Weather: NOT ATTACHED")

run_period = model.getRunPeriod()

print(
    "Run Period:",
    run_period.getBeginMonth(),
    run_period.getBeginDayOfMonth(),
    "to",
    run_period.getEndMonth(),
    run_period.getEndDayOfMonth()
)

simulation_control = model.getSimulationControl()

print(
    "Weather File Run:",
    simulation_control.runSimulationforWeatherFileRunPeriods()
)

print(
    "Sizing Run:",
    simulation_control.runSimulationforSizingPeriods()
)