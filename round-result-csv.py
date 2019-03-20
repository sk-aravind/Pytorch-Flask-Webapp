import pandas as pd

dataset = pd.read_csv('results.csv')

def try_cutoff(x):

    try:
        return round(float(x), 6)
    except Exception:
        return x

for field in dataset.columns:

    dataset[field] = dataset[field].map(try_cutoff)

# write new dataset result to CSV file
dataset.to_csv("results-6dp.csv", index = False)