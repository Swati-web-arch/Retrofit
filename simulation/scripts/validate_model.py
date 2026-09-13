import openstudio

model_path = (
    r"C:\Users\Lenovo\sttttttorage\Documents"
    r"\Retrofit\simulation\RetrofitIQ_baseline.osm"
)

translator = openstudio.osversion.VersionTranslator()
model = translator.loadModel(model_path)

if model.empty():
    print("MODEL LOAD FAILED")
else:
    model = model.get()

    print("MODEL LOAD SUCCESS")
    print("Spaces:", len(model.getSpaces()))
    print("Thermal Zones:", len(model.getThermalZones()))
    print("Ideal Loads Systems:", len(model.getZoneHVACIdealLoadsAirSystems()))