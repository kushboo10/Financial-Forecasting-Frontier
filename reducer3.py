import sys
from collections import defaultdict

contact_counts = defaultdict(lambda: {"yes": 0, "no": 0})

for line in sys.stdin:
    try:
        month, status = line.strip().split('\t')

        if status == "yes":
            contact_counts[month]["yes"] += 1
        else:
            contact_counts[month]["no"] += 1

    except:
        continue

print("Monthly Contacts and Subscription Status:\n")

for month in contact_counts:
    yes_count = contact_counts[month]["yes"]
    no_count = contact_counts[month]["no"]
    total = yes_count + no_count

    print(f"{month}: Total={total}, Subscribed={yes_count}, Not Subscribed={no_count}")