import pandas as pd

# Read dataset
df = pd.read_csv("hdfs/banking_data/bank.csv")

# Output poutcome and duration
for index, row in df.iterrows():
    print(f"{row['poutcome']}\t{row['duration']}")