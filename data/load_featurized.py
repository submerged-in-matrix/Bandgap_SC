from src.modules import *
# Load from CSV
X = pd.read_csv('Featurized_composition_Data.csv')
y = pd.read_csv('Featurized_Band_Gap_Data.csv')
y = y['gap expt'].tolist()

print(X.shape)
print(len(y))