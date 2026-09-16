import pandas as pd

from .monitoring_engine import MonitoringEngine


class RealtimeSimulator:
    def __init__(self, data=None, engine=None, start_index=0):
        self.engine = engine or MonitoringEngine()

        if data is None:
            self.engine.load()
            data = self.engine.data

        self.data = data.reset_index(drop=True)
        self.current_index = start_index

    def has_next(self):
        return self.current_index < len(self.data)

    def next_reading(self):
        # Nothing left to replay
        if not self.has_next():
            return None

        # Keep moving until we find a valid telemetry reading
        while self.has_next():
            row = self.data.iloc[self.current_index]
            self.current_index += 1

            # Skip rows where actual power is missing
            if "total_kw" not in row.index:
                continue

            if pd.isna(row["total_kw"]):
                continue

            return self.engine.analyze_row(row)

        # All remaining rows were invalid/missing
        return None

    def reset(self):
        self.current_index = 0
        self.engine.trend_monitor.reset()

    def get_progress(self):
        total = len(self.data)

        if total == 0:
            return 0.0

        return self.current_index / total