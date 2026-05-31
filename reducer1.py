import sys
from collections import defaultdict

job_totals = defaultdict(float)
job_counts = defaultdict(int)

for line in sys.stdin:
    try:
        job, balance = line.strip().split('\t')

        job_totals[job] += float(balance)
        job_counts[job] += 1

    except:
        continue

print("Average Balance by Job Type:\n")

for job in job_totals:
    average = job_totals[job] / job_counts[job]
    print(f"{job}: {average:.2f}")