import openstudio
from pathlib import Path

BASE_DIR = Path.cwd()

INPUT_FILE = BASE_DIR / "simulation" / "RetrofitIQ_baseline_hvac.osm"
OUTPUT_FILE = BASE_DIR / "simulation" / "RetrofitIQ_baseline_hvac_fixed.osm"

print("=" * 50)
print("FIXING SPACE SURFACE ORIENTATION")
print("=" * 50)

model = openstudio.model.Model.load(str(INPUT_FILE)).get()

print("Model loaded.")
print("Spaces:", len(model.getSpaces()))

for space in model.getSpaces():

    print()
    print("Checking:", space.nameString())

    # Let OpenStudio correct the surface orientations
    result = space.fixSurfacesWithIncorrectOrientation()

    print("Orientation fix result:", result)

print()
print("Checking space volumes...")

for space in model.getSpaces():

    print(
        space.nameString(),
        "Volume =",
        space.volume(),
        "m3"
    )

model.save(str(OUTPUT_FILE), True)

print()
print("=" * 50)
print("GEOMETRY FIX COMPLETE")
print("=" * 50)
print("Output:", OUTPUT_FILE)
print("=" * 50)