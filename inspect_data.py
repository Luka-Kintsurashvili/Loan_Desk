import sys
import pandas as pd

df = pd.read_csv(sys.argv[1], nrows=100000, low_memory=False)
print("rows x cols:", df.shape)
info = pd.DataFrame({
    "dtype": df.dtypes.astype(str),
    "missing_%": (df.isna().mean() * 100).round(1),
    "unique": df.nunique(),
    "example": df.iloc[0].astype(str).str[:20],
})
print(info.to_string())
for c in df.columns:
    if df[c].nunique() <= 4:
        print("\n" + c)
        print(df[c].value_counts(dropna=False).to_string())
