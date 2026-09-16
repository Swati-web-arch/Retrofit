from .config import (
    WATCH_DEVIATION,
    WARNING_DEVIATION,
    CRITICAL_DEVIATION,
)


class AnomalyDetector:

    def __init__(self):
        self.watch_threshold = WATCH_DEVIATION
        self.warning_threshold = WARNING_DEVIATION
        self.critical_threshold = CRITICAL_DEVIATION

    def calculate_deviation(self, actual, expected):

        if expected == 0:
            return 0.0

        deviation = (
            (actual - expected)
            / abs(expected)
        ) * 100

        return float(deviation)

    def classify(self, deviation):

        # Only positive deviation represents
        # excess energy consumption.
        if deviation >= self.critical_threshold:
            return "CRITICAL"

        if deviation >= self.warning_threshold:
            return "WARNING"

        if deviation >= self.watch_threshold:
            return "WATCH"

        if deviation <= -self.watch_threshold:
            return "UNDER_EXPECTED"

        return "NORMAL"

    def analyze(self, actual, expected):

        deviation = self.calculate_deviation(
            actual,
            expected
        )

        status = self.classify(deviation)

        return {
            "actual": float(actual),
            "expected": float(expected),
            "deviation_percent": round(
                deviation,
                2
            ),
            "status": status
        }