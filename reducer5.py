import sys
from collections import defaultdict

balance_totals = defaultdict(int)
balance_counts = defaultdict(int)

for line in sys.stdin:
    try:
        age_group, balance = line.strip().split('\t')

        balance_totals[age_group] += int(balance)
        balance_counts[age_group] += 1

    except:
        continue

print("Average Balance by Age Group:\n")

for group in balance_totals:
    average = balance_totals[group] / balance_counts[group]

    print(f"{group}: {average:.2f}")