from src.modules import *

df = load_dataset("matbench_expt_gap")  # Experimental bandgap dat
df.to_csv("matbench_expt_gap.csv", index=False)  

df = df.dropna(subset=["composition", "gap expt"])  # Drop NaNs
df["composition"] = df["composition"].apply(Composition)

print(df.columns)
print(df.head(3))
print("Sample compositions:")
print(df["composition"].head())
print("Type check:")
print(type(df["composition"].iloc[0]))