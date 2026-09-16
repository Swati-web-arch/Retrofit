from prediction.trend_monitor import TrendMonitor


monitor = TrendMonitor(
    window_size=5,
    persistence_count=3
)

test_deviations = [
    5,
    12,
    15,
    23,
    27,
    39
]

for deviation in test_deviations:

    result = monitor.add_reading(deviation)

    print(
        f"Deviation: {deviation:>5.1f}% | "
        f"Status: {result['status']:<8} | "
        f"Persistent: {result['persistent']} | "
        f"Consecutive: {result['consecutive_abnormal']}"
    )