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

print("\nPERSISTENCE TEST")
print("----------------")

# Reset previous trend history
engine.trend_monitor.reset()

# Select a continuous period from the real telemetry
start = "2018-02-07 06:00:00"
end = "2018-02-07 10:00:00"

period = engine.data[
    (engine.data["date"] >= start)
    & (engine.data["date"] < end)
]

for _, row in period.iterrows():

    result = engine.analyze_row(row)

    print(
        f"\nTime       : {result['timestamp']}"
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
        f"Consecutive: "
        f"{result['consecutive_abnormal']}"
    )

    print(
        f"Excess Cost: "
        f"₹{result['excess_cost_inr']}"
    )

    print(
        f"Alert      : "
        f"{result['alert']['title']}"
    )