from pathlib import Path
import pandas as pd

def inspect_data(path="data/interim/mobility_parsed.csv"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.main parse"
        )

    df = pd.read_csv(path)

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Unique vehicles:", df["vehicle_id"].nunique())
    print("Time range:", df["time"].min(), "to", df["time"].max())
    print("\nMissing values:")
    print(df.isna().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    print("\nRecords per vehicle:")
    print(df.groupby("vehicle_id").size().describe())
    print("\nRecords per time:")
    print(df.groupby("time").size().describe())

    return df
