from prediction.monitoring_engine import MonitoringEngine


engine = MonitoringEngine()

print("Loading data...")
engine.load()

print("Training model...")
metrics = engine.train()

print("\nMODEL")
print("-----")
print(f"MAE: {metrics['mae']:.2f} kW")
print(f"R² : {metrics['r2']:.3f}")

print("\nSEARCHING FOR EXCESS CONSUMPTION")
print("--------------------------------")

engine.trend_monitor.reset()

# Use a limited sample so the test finishes quickly
sample = engine.data.sample(
    n=min(500, len(engine.data)),
    random_state=42
)

found = 0
checked = 0
skipped = 0

for _, row in sample.iterrows():

    result = engine.analyze_row(row)

    # Skip missing actual power readings
    if result["status"] == "NO_DATA":
        skipped += 1
        continue

    checked += 1

    deviation = result["power_deviation_percent"]

    # Safety check
    if deviation is None:
        skipped += 1
        continue

    if deviation >= 10:

        print(
            f"\nTime       : "
            f"{result['timestamp']}"
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
            f"{deviation}%"
        )

        print(
            f"Status     : "
            f"{result['status']}"
        )

        print(
            f"Excess Cost: "
            f"₹{result['excess_cost_inr']}"
        )

        print(
            f"Alert      : "
            f"{result['alert']['title']}"
        )

        found += 1

        if found >= 10:
            break


print("\n--------------------------------")
print(f"Valid readings checked : {checked}")
print(f"Missing readings       : {skipped}")
print(f"Excess examples found  : {found}")