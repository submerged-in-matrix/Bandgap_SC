from src.modules import *
from data.load_raw import df

# --- Featurize Composition ---
featurizer = ElementProperty.from_preset("magpie")
df_feat = df.copy()
df_feat = featurizer.featurize_dataframe(df_feat, "composition", inplace=False)

# --- Prepare ML Dataset ---
X = df_feat[featurizer.feature_labels()]
y = df["gap expt"]

# X.to_csv('Featurized_composition_Data.csv', index=False)
# y.to_csv('Featurized_Band_Gap_Data.csv', index=False)