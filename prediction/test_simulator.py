from prediction.monitoring_engine import MonitoringEngine
from prediction.realtime_simulator import RealtimeSimulator


print("Creating monitoring engine...")

engine = MonitoringEngine()

print("Loading telemetry data...")

engine.load()

print("Training prediction model...")

metrics = engine.train()

print("\nMODEL")
print("-----")
print(f"MAE: {metrics['mae']:.2f} kW")
print(f"R² : {metrics['r2']:.3f}")


print("\nSTARTING TELEMETRY REPLAY")
print("--------------------------")


simulator = RealtimeSimulator(
    engine=engine
)


for i in range(10):

    result = simulator.next_reading()

    if result is None:
        break

    print(
        f"\nReading {i + 1}"
    )

    print(
        f"Time       : {result['timestamp']}"
    )

    print(
        f"Actual     : "
        f"{result['actual_power_kw']} kW"
    )

    print(
        f"Expected   : "
        f"{result['expected_power_kw']} kW"
    )

    print(
        f"Deviation  : "
        f"{result['power_deviation_percent']}%"
    )

    print(
        f"Status     : "
        f"{result['status']}"
    )

    print(
        f"Persistent : "
        f"{result['persistent']}"
    )

    print(
        f"Excess Cost: "
        f"₹{result['excess_cost_inr']}"
    )

    print(
        f"Alert      : "
        f"{result['alert']['title']}"
    )