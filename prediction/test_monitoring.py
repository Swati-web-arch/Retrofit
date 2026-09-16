from prediction.monitoring_engine import MonitoringEngine


engine = MonitoringEngine()

print("Loading data...")
engine.load()

print("Training model...")
metrics = engine.train()

print("\nMODEL PERFORMANCE")
print("-----------------")
print(f"MAE: {metrics['mae']:.2f} kW")
print(f"R² : {metrics['r2']:.3f}")


print("\nANALYZING RECENT READINGS")
print("--------------------------")


recent_data = engine.data.tail(10)


for _, row in recent_data.iterrows():

    result = engine.analyze_row(row)

    print(
        f"\nTime: {result['timestamp']}"
    )

    print(
        f"Actual Power   : "
        f"{result['actual_power_kw']} kW"
    )

    print(
        f"Expected Power : "
        f"{result['expected_power_kw']} kW"
    )

    print(
        f"Deviation      : "
        f"{result['power_deviation_percent']}%"
    )

    print(
        f"Expected Cost  : "
        f"₹{result['expected_cost_inr']}"
    )

    print(
        f"Actual Cost    : "
        f"₹{result['actual_cost_inr']}"
    )

    print(
        f"Excess Cost    : "
        f"₹{result['excess_cost_inr']}"
    )

    print(
        f"Status         : "
        f"{result['status']}"
    )

    print(
        f"Persistent     : "
        f"{result['persistent']}"
    )

    print(
        f"Consecutive    : "
        f"{result['consecutive_abnormal']}"
    )

    print(
        f"Alert          : "
        f"{result['alert']['title']}"
    )