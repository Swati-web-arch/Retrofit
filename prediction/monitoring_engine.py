import pandas as pd

from .config import DEFAULT_TARIFF, MIN_ABNORMAL_READINGS
from .data_loader import prepare_data
from .predictor import EnergyPredictor
from .anomaly_detector import AnomalyDetector
from .alert_engine import AlertEngine
from .trend_monitor import TrendMonitor


class MonitoringEngine:

    def __init__(self, tariff=DEFAULT_TARIFF):

        self.tariff = tariff

        self.predictor = EnergyPredictor(
            tariff=tariff
        )

        self.detector = AnomalyDetector()

        self.alert_engine = AlertEngine()

        self.trend_monitor = TrendMonitor(
            window_size=5,
            persistence_count=MIN_ABNORMAL_READINGS
        )

        self.data = None
        self.trained = False

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------

    def load(self, path=None):

        if path is None:
            self.data = prepare_data()

        else:
            self.data = prepare_data(path)

        return self.data

    # --------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------

    def train(self):

        if self.data is None:
            self.load()

        metrics = self.predictor.train(
            self.data
        )

        self.trained = True

        return metrics

    # --------------------------------------------------
    # ANALYZE ONE READING
    # --------------------------------------------------

    def analyze_row(
        self,
        row,
        component="HVAC System"
    ):

        if not self.trained:

            raise RuntimeError(
                "Monitoring engine must be trained first."
            )

        # Convert the row into a one-row DataFrame
        if isinstance(row, pd.Series):

            row_df = row.to_frame().T

        else:

            row_df = pd.DataFrame([row])

        # --------------------------------------------------
        # CHECK ACTUAL POWER FIRST
        # --------------------------------------------------

        if "total_kw" not in row_df.columns:

            return self._missing_reading_result(
                row_df,
                component
            )

        actual_power = row_df["total_kw"].iloc[0]

        # Missing actual reading
        if pd.isna(actual_power):

            return self._missing_reading_result(
                row_df,
                component
            )

        # Convert to normal Python float
        actual_power = float(actual_power)

        # --------------------------------------------------
        # PREDICT EXPECTED POWER
        # --------------------------------------------------

        predicted_power = self.predictor.predict_power(
            row_df
        )[0]

        predicted_power = float(
            predicted_power
        )

        # --------------------------------------------------
        # ANOMALY ANALYSIS
        # --------------------------------------------------

        analysis = self.detector.analyze(
            actual=actual_power,
            expected=predicted_power
        )

        deviation = analysis[
            "deviation_percent"
        ]

        # --------------------------------------------------
        # TREND MONITORING
        # --------------------------------------------------

        trend = self.trend_monitor.add_reading(
            deviation
        )

        analysis["status"] = trend[
            "status"
        ]

        # --------------------------------------------------
        # ENERGY + COST
        # --------------------------------------------------

        energy, cost = self.predictor.predict_cost(
            predicted_power,
            hours=1
        )

        expected_energy = float(
            energy
        )

        expected_cost = float(
            cost
        )

        actual_cost = (
            actual_power
            * self.tariff
        )

        excess_cost = (
            actual_cost
            - expected_cost
        )

        # --------------------------------------------------
        # ALERT
        # --------------------------------------------------

        alert = self.alert_engine.generate_alert(
            analysis,
            component=component
        )

        # Add persistence information
        if trend["persistent"]:

            alert["message"] += (
                f" This deviation has persisted for "
                f"{trend['consecutive_abnormal']} "
                f"consecutive readings."
            )

            alert["action"] = (
                "Persistent abnormal behavior detected. "
                "Inspect the relevant equipment and "
                "operating conditions."
            )

        # --------------------------------------------------
        # TIMESTAMP
        # --------------------------------------------------

        timestamp = None

        if "date" in row_df.columns:

            timestamp = row_df[
                "date"
            ].iloc[0]

        # --------------------------------------------------
        # RETURN RESULT
        # --------------------------------------------------

        return {

            "timestamp": timestamp,

            "actual_power_kw": round(
                actual_power,
                2
            ),

            "expected_power_kw": round(
                predicted_power,
                2
            ),

            "power_deviation_percent": round(
                deviation,
                2
            ),

            "expected_energy_kwh": round(
                expected_energy,
                2
            ),

            "expected_cost_inr": round(
                expected_cost,
                2
            ),

            "actual_cost_inr": round(
                actual_cost,
                2
            ),

            "excess_cost_inr": round(
                excess_cost,
                2
            ),

            "status": trend[
                "status"
            ],

            "persistent": trend[
                "persistent"
            ],

            "consecutive_abnormal": trend[
                "consecutive_abnormal"
            ],

            "average_deviation_percent": trend[
                "average_deviation"
            ],

            "alert": alert
        }

    # --------------------------------------------------
    # MISSING READING RESULT
    # --------------------------------------------------

    def _missing_reading_result(
        self,
        row_df,
        component
    ):

        timestamp = None

        if "date" in row_df.columns:

            timestamp = row_df[
                "date"
            ].iloc[0]

        return {

            "timestamp": timestamp,

            "actual_power_kw": None,

            "expected_power_kw": None,

            "power_deviation_percent": None,

            "expected_energy_kwh": None,

            "expected_cost_inr": None,

            "actual_cost_inr": None,

            "excess_cost_inr": None,

            "status": "NO_DATA",

            "persistent": False,

            "consecutive_abnormal": 0,

            "average_deviation_percent": 0.0,

            "alert": {

                "severity": "INFO",

                "title": (
                    f"{component} — Missing Reading"
                ),

                "message": (
                    "Actual energy consumption data "
                    "is unavailable for this timestamp."
                ),

                "action": (
                    "Check the telemetry source "
                    "or sensor connection."
                )
            }
        }

    # --------------------------------------------------
    # ANALYZE HISTORY
    # --------------------------------------------------

    def analyze_history(
        self,
        df=None,
        component="HVAC System"
    ):

        if df is None:

            df = self.data

        results = []

        for _, row in df.iterrows():

            try:

                result = self.analyze_row(
                    row,
                    component
                )

                results.append(result)

            except Exception as e:

                print(
                    f"Skipping row because of error: {e}"
                )

        return pd.DataFrame(
            results
        )