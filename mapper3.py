import pandas as pd

# Read dataset
df = pd.read_csv("hdfs/banking_data/bank.csv")

# Output month and subscription status
for index, row in df.iterrows():
    print(f"{row['month']}\t{row['y']}")