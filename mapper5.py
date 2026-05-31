import pandas as pd

# Read dataset
df = pd.read_csv("hdfs/banking_data/bank.csv")

# Create age groups
for index, row in df.iterrows():

    age = row['age']
    balance = row['balance']

    # Categorize ages
    if age < 30:
        group = "Young"
    elif age < 50:
        group = "Middle-Aged"
    else:
        group = "Senior"

    print(f"{group}\t{balance}")