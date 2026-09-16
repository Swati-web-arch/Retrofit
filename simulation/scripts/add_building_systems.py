import openstudio
from pathlib import Path

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path.cwd()

INPUT_OSM = BASE_DIR / "simulation" / "RetrofitIQ_test.osm"
OUTPUT_OSM = BASE_DIR / "simulation" / "RetrofitIQ_baseline.osm"

# =========================================================
# LOAD MODEL
# =========================================================

model = openstudio.model.Model.load(str(INPUT_OSM)).get()

print("MODEL LOADED")
print("Input:", INPUT_OSM)

# =========================================================
# 1. BASIC CONSTRUCTION
# =========================================================

material = openstudio.model.StandardOpaqueMaterial(model)

material.setName("Basic Wall Material")
material.setRoughness("MediumRough")
material.setThickness(0.20)
material.setConductivity(1.0)
material.setDensity(1800)
material.setSpecificHeat(800)

construction = openstudio.model.Construction(model)

construction.setName("Basic Construction")
construction.insertLayer(0, material)

# Assign construction to every surface
for surface in model.getSurfaces():
    surface.setConstruction(construction)

# =========================================================
# 2. ALWAYS-ON FRACTION SCHEDULE
# =========================================================

fraction_limits = openstudio.model.ScheduleTypeLimits(model)

fraction_limits.setName("Fraction Schedule Limits")
fraction_limits.setLowerLimitValue(0.0)
fraction_limits.setUpperLimitValue(1.0)
fraction_limits.setNumericType("Continuous")

always_on = openstudio.model.ScheduleConstant(model)

always_on.setName("Always On")
always_on.setValue(1.0)
always_on.setScheduleTypeLimits(fraction_limits)

# =========================================================
# 3. ACTIVITY LEVEL SCHEDULE
# =========================================================

activity_limits = openstudio.model.ScheduleTypeLimits(model)

activity_limits.setName("Activity Level Schedule Limits")
activity_limits.setLowerLimitValue(0.0)
activity_limits.setUpperLimitValue(1000.0)
activity_limits.setNumericType("Continuous")
activity_limits.setUnitType("ActivityLevel")

activity_schedule = openstudio.model.ScheduleConstant(model)

activity_schedule.setName("Office Activity Level")
activity_schedule.setValue(120.0)
activity_schedule.setScheduleTypeLimits(activity_limits)

# 120 W/person is a reasonable simple office activity assumption.

# =========================================================
# 4. OCCUPANCY
# =========================================================

people_def = openstudio.model.PeopleDefinition(model)

people_def.setName("Office Occupancy Definition")

for space in model.getSpaces():

    people = openstudio.model.People(people_def)

    people.setName(
        "Office Occupancy - " + space.nameString()
    )

    # Number of occupants
    people.setNumberofPeopleSchedule(always_on)

    # Activity level: 120 W/person
    people.setActivityLevelSchedule(activity_schedule)

    people.setSpace(space)

# =========================================================
# 5. LIGHTING
# =========================================================

lights_def = openstudio.model.LightsDefinition(model)

lights_def.setName("Office Lights Definition")

lights_def.setWattsperSpaceFloorArea(10.0)

for space in model.getSpaces():

    lights = openstudio.model.Lights(lights_def)

    lights.setName(
        "Office Lights - " + space.nameString()
    )

    lights.setSchedule(always_on)
    lights.setSpace(space)

# =========================================================
# 6. ELECTRIC EQUIPMENT
# =========================================================

equipment_def = (
    openstudio.model.ElectricEquipmentDefinition(model)
)

equipment_def.setName(
    "Office Equipment Definition"
)

equipment_def.setWattsperSpaceFloorArea(8.0)

for space in model.getSpaces():

    equipment = openstudio.model.ElectricEquipment(
        equipment_def
    )

    equipment.setName(
        "Office Equipment - " + space.nameString()
    )

    equipment.setSchedule(always_on)
    equipment.setSpace(space)

# =========================================================
# 7. CREATE THERMAL ZONES
# =========================================================

# Remove existing thermal-zone assignments
for space in model.getSpaces():

    if space.thermalZone().is_initialized():
        space.resetThermalZone()

# Create one thermal zone for each space
for space in model.getSpaces():

    zone = openstudio.model.ThermalZone(model)

    zone.setName(
        space.nameString() + " Thermal Zone"
    )

    space.setThermalZone(zone)

# =========================================================
# 8. IDEAL LOADS HVAC
# =========================================================

for zone in model.getThermalZones():

    hvac = openstudio.model.ZoneHVACIdealLoadsAirSystem(
        model
    )

    hvac.setName(
        "Ideal Loads - " + zone.nameString()
    )

    hvac.addToThermalZone(zone)

# =========================================================
# 9. SAVE MODEL
# =========================================================

model.save(
    str(OUTPUT_OSM),
    True
)

# =========================================================
# 10. FINAL CHECK
# =========================================================

print("")
print("========================================")
print("BASELINE MODEL CREATED")
print("========================================")

print("Output:", OUTPUT_OSM)

print(
    "Spaces:",
    len(model.getSpaces())
)

print(
    "Thermal Zones:",
    len(model.getThermalZones())
)

print(
    "Ideal Loads Systems:",
    len(model.getZoneHVACIdealLoadsAirSystems())
)

print("========================================")