from prediction.monitoring_engine import MonitoringEngine


engine = MonitoringEngine()

print("Loading data...")
engine.load()

print("\nDATASET INFO")
print("------------")

print("Rows:", len(engine.data))
print("Columns:")
print(engine.data.columns.tolist())

print("\nTOTAL_KW INFO")
print("-------------")

print(engine.data["total_kw"].head(10))

print("\nMissing total_kw:")
print(engine.data["total_kw"].isna().sum())

print("\nNon-missing total_kw:")
print(engine.data["total_kw"].notna().sum())

print("\nFirst 10 rows:")
print(
    engine.data[
        ["date", "total_kw"]
    ].head(10)
)