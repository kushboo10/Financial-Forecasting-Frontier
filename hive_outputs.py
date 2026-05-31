import pandas as pd

# Load dataset
df = pd.read_csv("data/bank.csv")

print("\nTOTAL CLIENTS")
print(df.shape[0])

print("\nFIRST 10 ROWS")
print(df.head(10))

print("\nMARRIED CLIENTS WITH PERSONAL LOAN")
print(df[(df['marital'] == 'married') & (df['loan'] == 'yes')].head())

print("\nTOP 10 HIGHEST BALANCE CLIENTS")
print(df[['job', 'marital', 'balance']]
      .sort_values(by='balance', ascending=False)
      .head(10))

print("\nAVERAGE AGE BY JOB")
print(df.groupby('job')['age'].mean())

print("\nDEFAULT COUNT BY EDUCATION")
print(df[df['default'] == 'yes']
      .groupby('education')
      .size())

print("\nTOP JOBS WITH HIGHEST AVERAGE BALANCE")
top_jobs = df.groupby('job').agg({
    'balance': 'mean',
    'y': lambda x: (x == 'yes').mean() * 100
})

print(top_jobs.sort_values(by='balance', ascending=False).head(5))

print("\nMONTH WITH HIGHEST CONTACTS")

month_analysis = df.groupby('month').agg({
    'y': lambda x: (x == 'yes').mean() * 100
})

month_analysis['total_contacts'] = df.groupby('month').size()

print(month_analysis.sort_values(by='total_contacts', ascending=False).head(1))

print("\nCORRELATION BETWEEN AGE AND BALANCE")
print(df['age'].corr(df['balance']))

print("\nAVERAGE BALANCE BY EDUCATION")
print(df.groupby('education')['balance'].mean())

print("\nSUBSCRIPTION RATE BY POUTCOME")
print(df.groupby('poutcome')['y']
      .apply(lambda x: (x == 'yes').mean() * 100))

print("\nAVERAGE CONTACT DURATION")
print(df.groupby('y')['duration'].mean())