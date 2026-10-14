
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent

files = {
    "Original dataset": root / "data" / "raw" / "ai4i2020.csv",
    "Cleaned dataset": root / "data" / "processed" / "ai4i2020_cleaned.csv",
    "Training dataset": root / "data" / "processed" / "train.csv",
    "Test dataset": root / "data" / "processed" / "test.csv",
}

for name, path in files.items():
    print(f"\n{'=' * 55}")
    print(name)

    if not path.exists():
        print("File not found:", path)
        continue

    df = pd.read_csv(path)

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    if "Machine failure" in df.columns:
        print("Failure distribution:")
        print(df["Machine failure"].value_counts().sort_index())

    print("Missing values:", int(df.isnull().sum().sum()))
