import sys
from collections import defaultdict

housing_counts = defaultdict(lambda: {"yes": 0, "no": 0})

for line in sys.stdin:
    try:
        education, housing = line.strip().split('\t')

        if housing == "yes":
            housing_counts[education]["yes"] += 1
        else:
            housing_counts[education]["no"] += 1

    except:
        continue

print("Housing Loan Count by Education Level:\n")

for education in housing_counts:
    yes_count = housing_counts[education]["yes"]
    no_count = housing_counts[education]["no"]

    print(f"{education}: Yes={yes_count}, No={no_count}")