import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from .config import (
    TARGET_COLUMN,
    TIMESTAMP_COLUMN,
    RANDOM_STATE,
    DEFAULT_TARIFF,
)


class EnergyPredictor:

    def __init__(self, tariff=DEFAULT_TARIFF):

        self.tariff = tariff

        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=1
        )

        self.feature_columns = []
        self.is_trained = False

        self.metrics = {}


    def create_features(self, df):

        data = df.copy()

        # -------------------------------------------------
        # TIME FEATURES
        # -------------------------------------------------

        if TIMESTAMP_COLUMN in data.columns:

            timestamp = pd.to_datetime(
                data[TIMESTAMP_COLUMN],
                errors="coerce"
            )

            data["hour"] = timestamp.dt.hour
            data["day_of_week"] = timestamp.dt.dayofweek
            data["month"] = timestamp.dt.month
            data["day_of_month"] = timestamp.dt.day

        # -------------------------------------------------
        # EXISTING DATASET FEATURES
        # -------------------------------------------------

        preferred_features = [
            # HVAC measurements
            "hvac_N",
            "hvac_S",

        # Indoor conditions
            "zone_temp_avg",
            "air_temp_set_1",
            "air_temp_set_2",
            "relative_humidity_set_1",

        # RTU measurements
            "rtu_sa_temp_avg",
            "rtu_oa_damper_avg",
            "rtu_ma_temp_avg",
            "rtu_econ_sp_avg",

        # Occupancy / environment
            "total_occ",
            "outdoor_temp_f",
            "indoor_temp_f",

        # Time features
            "hour",
            "day_of_week",
            "month",
            "day_of_month",
            "is_business_hours",
        ]

        available = [
            col for col in preferred_features
            if col in data.columns
        ]

        self.feature_columns = available

        return data


    def train(self, df):

        data = self.create_features(df)

        # Make sure target exists
        if TARGET_COLUMN not in data.columns:
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' "
                f"not found in dataset."
            )

        # Keep only rows where target exists
        data = data.dropna(
            subset=[TARGET_COLUMN]
        )

        X = data[self.feature_columns].copy()
        y = data[TARGET_COLUMN].copy()

        # Fill missing feature values
        X = X.fillna(X.median(numeric_only=True))

        # Time-based split
        split_index = int(len(data) * 0.80)

        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]

        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        self.model.fit(
            X_train,
            y_train
        )

        predictions = self.model.predict(X_test)

        self.metrics = {
            "mae": float(
                mean_absolute_error(
                    y_test,
                    predictions
                )
            ),
            "r2": float(
                r2_score(
                    y_test,
                    predictions
                )
            )
        }

        self.is_trained = True

        return self.metrics


    def predict_power(self, df):

        if not self.is_trained:
            raise RuntimeError(
                "Model must be trained before prediction."
            )

        data = self.create_features(df)

        X = data[self.feature_columns].copy()

        X = X.fillna(
            X.median(numeric_only=True)
        )

        predictions = self.model.predict(X)

        return predictions


    def predict_power(self, df):
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction.")

        data = self.create_features(df)
        X = data[self.feature_columns].copy()
        X = X.fillna(X.median(numeric_only=True))

        predictions = self.model.predict(X)

        return predictions

    def predict_cost(self, predicted_power, hours=1):
        predicted_energy = np.asarray(predicted_power) * hours
        predicted_cost = predicted_energy * self.tariff

        return predicted_energy, predicted_cost