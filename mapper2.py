import pandas as pd

# Read dataset
df = pd.read_csv("hdfs/banking_data/bank.csv")

# Print education and housing loan status
for index, row in df.iterrows():
    print(f"{row['education']}\t{row['housing']}")