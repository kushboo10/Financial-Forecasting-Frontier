import pandas as pd

# Read CSV 
df = pd.read_csv("hdfs/banking_data/bank.csv")

# Print job and balance
for index, row in df.iterrows():
    print(f"{row['job']}\t{row['balance']}")