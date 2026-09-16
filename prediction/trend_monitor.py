from collections import deque


class TrendMonitor:
    """
    Tracks recent energy deviations and determines
    whether excess energy consumption is persistent.
    """

    def __init__(
        self,
        window_size=5,
        persistence_count=3,
        watch_threshold=10.0,
        warning_threshold=20.0,
        critical_threshold=35.0
    ):

        self.window_size = window_size
        self.persistence_count = persistence_count

        self.watch_threshold = watch_threshold
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

        self.deviation_history = deque(
            maxlen=window_size
        )

    def add_reading(self, deviation):

        self.deviation_history.append(
            float(deviation)
        )

        return self.get_status()

    def get_status(self):

        if not self.deviation_history:

            return {
                "status": "NORMAL",
                "persistent": False,
                "consecutive_abnormal": 0,
                "average_deviation": 0.0
            }

        recent = list(
            self.deviation_history
        )

        # Only positive deviations represent
        # excess energy consumption.
        consecutive_abnormal = 0

        for deviation in reversed(recent):

            if deviation >= self.watch_threshold:
                consecutive_abnormal += 1
            else:
                break

        average_deviation = (
            sum(recent) / len(recent)
        )

        persistent = (
            consecutive_abnormal
            >= self.persistence_count
        )

        latest_deviation = recent[-1]

        if latest_deviation >= self.critical_threshold:

            status = "CRITICAL"

        elif latest_deviation >= self.warning_threshold:

            status = "WARNING"

        elif latest_deviation >= self.watch_threshold:

            status = "WATCH"

        elif latest_deviation <= -self.watch_threshold:

            status = "UNDER_EXPECTED"

        else:

            status = "NORMAL"

        return {
            "status": status,
            "persistent": persistent,
            "consecutive_abnormal": consecutive_abnormal,
            "average_deviation": round(
                average_deviation,
                2
            )
        }

    def reset(self):

        self.deviation_history.clear()