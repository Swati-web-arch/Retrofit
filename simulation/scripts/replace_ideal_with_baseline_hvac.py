import openstudio
from pathlib import Path

print("=" * 50)
print("CREATING BASELINE HVAC")
print("=" * 50)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = Path(r"C:\Users\Lenovo\sttttttorage\Documents\Retrofit")

INPUT_MODEL = BASE_DIR / "simulation" / "RetrofitIQ_ready.osm"
OUTPUT_MODEL = BASE_DIR / "simulation" / "RetrofitIQ_baseline_hvac_fixed.osm"

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

loaded_model = openstudio.model.Model.load(str(INPUT_MODEL))

if loaded_model.empty():
    raise RuntimeError("Failed to load OpenStudio model.")

model = loaded_model.get()

print("Model loaded.")
print(f"Spaces: {len(model.getSpaces())}")
print(f"Zones: {len(model.getThermalZones())}")

# --------------------------------------------------
# REMOVE IDEAL LOADS
# --------------------------------------------------

ideal_loads = model.getZoneHVACIdealLoadsAirSystems()

print(f"Existing Ideal Loads: {len(ideal_loads)}")

for ideal in ideal_loads:
    print(f"Removing: {ideal.nameString()}")
    ideal.remove()

print("Ideal Loads removed.")

# --------------------------------------------------
# ALWAYS ON SCHEDULE
# --------------------------------------------------

always_on = model.alwaysOnDiscreteSchedule()

# --------------------------------------------------
# CREATE AIR LOOP
# --------------------------------------------------

air_loop = openstudio.model.AirLoopHVAC(model)
air_loop.setName("Baseline Constant Volume AHU")
# Explicitly fix the AirLoopHVAC design supply airflow.
# This prevents EnergyPlus from treating the air loop as autosized.
air_loop.setDesignSupplyAirFlowRate(1.0)

print("AirLoop design supply airflow = 1.0 m3/s")
# --------------------------------------------------
# FIX AIR LOOP SIZING VALUES
# --------------------------------------------------

sizing = air_loop.sizingSystem()

sizing.setCoolingDesignCapacityMethod("CoolingDesignCapacity")
sizing.setCoolingDesignCapacity(60000.0)

sizing.setHeatingDesignCapacityMethod("HeatingDesignCapacity")
sizing.setHeatingDesignCapacity(0.0)

sizing.setCoolingDesignAirFlowMethod("DesignDay")
sizing.setCoolingDesignAirFlowRate(1.0)

sizing.setHeatingDesignAirFlowMethod("DesignDay")
sizing.setHeatingDesignAirFlowRate(1.0)

print("Air-loop sizing values fixed.")
print("Cooling design capacity = 60 kW")
print("Heating design capacity = 0 kW")
print("Design airflow = 1.0 m3/s")

# IMPORTANT:
# NO AirLoopHVAC sizing object is created.
# All equipment has fixed values.

# --------------------------------------------------
# CONSTANT VOLUME FAN
# --------------------------------------------------

fan = openstudio.model.FanConstantVolume(
    model,
    always_on
)

fan.setName("Baseline Constant Speed Supply Fan")

fan.setMaximumFlowRate(1.0)
fan.setFanEfficiency(0.70)
fan.setPressureRise(500.0)
fan.setMotorEfficiency(0.90)

print("Constant-volume fan added.")
print("Fan airflow = 1.0 m3/s")

# --------------------------------------------------
# DX COOLING COIL
# --------------------------------------------------

cooling_coil = openstudio.model.CoilCoolingDXSingleSpeed(model)

cooling_coil.setName("Baseline DX Cooling Coil")

cooling_coil.setRatedTotalCoolingCapacity(60000.0)
cooling_coil.setRatedAirFlowRate(1.0)
cooling_coil.setRatedCOP(3.0)
cooling_coil.setRatedSensibleHeatRatio(0.75)
print("DX cooling coil added.")
print("Cooling capacity = 60 kW")
print("Cooling airflow = 1.0 m3/s")
print("COP = 3.0")

# --------------------------------------------------
# SUPPLY AIR TEMPERATURE
# --------------------------------------------------

supply_temp_schedule = openstudio.model.ScheduleRuleset(model)
supply_temp_schedule.setName("Baseline Supply Air Temperature")

supply_temp_schedule.defaultDaySchedule().addValue(
    openstudio.Time(0, 0, 0),
    13.0
)

setpoint_manager = openstudio.model.SetpointManagerScheduled(
    model,
    supply_temp_schedule
)

setpoint_manager.setName(
    "Baseline Supply Air Setpoint Manager"
)

print("Supply air setpoint = 13 C")

# --------------------------------------------------
# ADD FAN AND COIL TO AIR LOOP
# --------------------------------------------------

# Connect the DX coil and fan using OpenStudio's node-based HVAC API.
#
# Do NOT use addBranchForHVACComponent() for the DX coil here.
# addToNode() creates the proper component/node connections.

cooling_added = cooling_coil.addToNode(
    air_loop.supplyOutletNode()
)

if not cooling_added:
    raise RuntimeError("Failed to add DX cooling coil to the air-loop supply node.")

fan_added = fan.addToNode(
    air_loop.supplyOutletNode()
)

if not fan_added:
    raise RuntimeError("Failed to add constant-volume fan to the air-loop supply node.")

setpoint_manager.addToNode(
    air_loop.supplyOutletNode()
)

print("DX cooling coil connected using addToNode().")
print("Constant-volume fan connected using addToNode().")


# --------------------------------------------------
# CONNECT ZONES
# --------------------------------------------------

zones = model.getThermalZones()

for zone in zones:

    print(f"Connecting: {zone.nameString()}")

    terminal = openstudio.model.AirTerminalSingleDuctVAVNoReheat(
        model,
        always_on
    )

    terminal.setName(
        "VAV Terminal - " + zone.nameString()
    )

    terminal.setMaximumAirFlowRate(0.5)


    air_loop.addBranchForZone(zone, terminal)

print("All zones connected.")

# --------------------------------------------------
# SIMULATION CONTROL
# --------------------------------------------------

simulation_control = model.getSimulationControl()

simulation_control.setDoZoneSizingCalculation(False)
simulation_control.setDoSystemSizingCalculation(False)
simulation_control.setDoPlantSizingCalculation(False)

simulation_control.setRunSimulationforSizingPeriods(False)
simulation_control.setRunSimulationforWeatherFileRunPeriods(True)

print("All sizing calculations disabled.")
print("Weather-file run: True")
print("Zone sizing calculation: False")
print("System sizing calculation: False")
print("Plant sizing calculation: False")
print("Sizing-period run: False")

# --------------------------------------------------
# SAVE
# --------------------------------------------------

model.save(
    str(OUTPUT_MODEL),
    True
)

print()
print("=" * 50)
print("BASELINE HVAC CREATED SUCCESSFULLY")
print("=" * 50)

print(f"Output: {OUTPUT_MODEL}")
print(f"Air Loops: {len(model.getAirLoopHVACs())}")
print(f"Constant Volume Fans: {len(model.getFanConstantVolumes())}")
print(f"DX Cooling Coils: {len(model.getCoilCoolingDXSingleSpeeds())}")
print(f"Ideal Loads: {len(model.getZoneHVACIdealLoadsAirSystems())}")

print("=" * 50)