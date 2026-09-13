import openstudio
from pathlib import Path

MODEL_FILE = Path.cwd() / "simulation" / "RetrofitIQ_mumbai.osm"

print("Loading Mumbai model...")

model = openstudio.model.Model.load(str(MODEL_FILE)).get()

print("MODEL LOAD SUCCESS")

weather = model.getOptionalWeatherFile()

if weather.is_initialized():
    wf = weather.get()
    print("WEATHER FILE ATTACHED: YES")
    print("City:", wf.city())
    print("Country:", wf.country())
    print("Latitude:", wf.latitude())
    print("Longitude:", wf.longitude())
else:
    print("WEATHER FILE ATTACHED: NO")