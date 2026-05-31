import sys
from collections import defaultdict

duration_totals = defaultdict(int)
duration_counts = defaultdict(int)

for line in sys.stdin:
    try:
        poutcome, duration = line.strip().split('\t')

        duration_totals[poutcome] += int(duration)
        duration_counts[poutcome] += 1

    except:
        continue

print("Average Contact Duration by Campaign Outcome:\n")

for poutcome in duration_totals:
    average = duration_totals[poutcome] / duration_counts[poutcome]

    print(f"{poutcome}: {average:.2f} seconds")