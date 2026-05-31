import os

# IMPORTANT FOR WINDOWS
os.environ["PYSPARK_PYTHON"] = r"C:\Users\kushb\AppData\Local\Programs\Python\Python311\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = r"C:\Users\kushb\AppData\Local\Programs\Python\Python311\python.exe"

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StringType
from pyspark.sql.functions import udf
import matplotlib.pyplot as plt

# CREATE SPARK SESSION
spark = SparkSession.builder \
    .appName("Banking Spark Processing") \
    .getOrCreate()

# LOAD CSV FILE
df = spark.read.csv(
    r"C:\Users\kushb\Banking_Project\data\bank.csv",
    header=True,
    inferSchema=True
)

# ======================================================
# 1. DATA LOADING AND BASIC INSPECTION
# ======================================================

print("\nFIRST 5 ROWS")
df.show(5)

print("\nSCHEMA")
df.printSchema()

print("\nSUMMARY")
df.describe().show()

# ======================================================
# 2. FILTERING AND COLUMN OPERATIONS
# ======================================================

print("\nCLIENTS WITH BALANCE > 1000")
high_balance = df.filter(df.balance > 1000)
high_balance.show(5)

# CREATE QUARTER COLUMN
df = df.withColumn(
    "quarter",
    when(col("month").isin("jan", "feb", "mar"), "Q1")
    .when(col("month").isin("apr", "may", "jun"), "Q2")
    .when(col("month").isin("jul", "aug", "sep"), "Q3")
    .otherwise("Q4")
)

print("\nMONTH AND QUARTER")
df.select("month", "quarter").show(10)

# ======================================================
# 3. GROUPBY AND AGGREGATION
# ======================================================

print("\nAVERAGE BALANCE & MEDIAN AGE BY JOB")

job_stats = df.groupBy("job").agg(
    avg("balance").alias("avg_balance"),
    expr("percentile(age, 0.5)").alias("median_age")
)

job_stats.show()

print("\nSUBSCRIBED CLIENTS BY MARITAL STATUS")

subscribed = df.filter(df.y == "yes") \
    .groupBy("marital") \
    .count()

subscribed.show()

# ======================================================
# 4. UDF AGE GROUPS
# ======================================================

def age_group(age):
    if age < 30:
        return "<30"
    elif age <= 60:
        return "30-60"
    else:
        return ">60"

age_udf = udf(age_group, StringType())

df = df.withColumn("age_group", age_udf(col("age")))

print("\nAGE GROUPS")
df.select("age", "age_group").show(10)

# ======================================================
# 5. ADVANCED DATA TRANSFORMATIONS
# ======================================================

print("\nSUBSCRIPTION RATE BY EDUCATION")

education_rate = df.groupBy("education").agg(
    (
        sum(when(col("y") == "yes", 1).otherwise(0))
        / count("*") * 100
    ).alias("subscription_rate")
)

education_rate.show()

print("\nTOP 3 PROFESSIONS WITH HIGHEST DEFAULT RATE")

default_rate = df.groupBy("job").agg(
    (
        sum(when(col("default") == "yes", 1).otherwise(0))
        / count("*") * 100
    ).alias("default_rate")
)

default_rate.orderBy(desc("default_rate")).show(3)

# ======================================================
# 6. STRING MANIPULATION
# ======================================================

df = df.withColumn(
    "job_marital",
    concat_ws("_", col("job"), col("marital"))
)

df = df.withColumn(
    "contact_upper",
    upper(col("contact"))
)

print("\nJOB + MARITAL")
df.select("job_marital").show(5)

print("\nCONTACT IN UPPERCASE")
df.select("contact_upper").show(5)

# ======================================================
# 7. DATA VISUALIZATION
# ======================================================

print("\nCREATING BAR CHART")

pandas_df = df.toPandas()

job_counts = pandas_df["job"].value_counts()

plt.figure(figsize=(10, 6))
job_counts.plot(kind='bar')
plt.title("Count of Clients by Job Type")
plt.xlabel("Job")
plt.ylabel("Count")
plt.xticks(rotation=45)

plt.show()

# ======================================================
# 8. COMPLEX QUERIES
# ======================================================

print("\nMONTH WITH HIGHEST CONTACTS")

month_analysis = df.groupBy("month").agg(
    count("*").alias("total_contacts"),
    (
        sum(when(col("y") == "yes", 1).otherwise(0))
        / count("*") * 100
    ).alias("success_rate")
)

month_analysis.orderBy(desc("total_contacts")).show(1)

print("\nAVERAGE DURATION FOR SUBSCRIBED VS NON-SUBSCRIBED")

duration_analysis = df.groupBy("y").agg(
    avg("duration").alias("avg_duration")
)

duration_analysis.show()

# ======================================================
# 9. CORRELATION
# ======================================================

print("\nCORRELATION BETWEEN AGE AND BALANCE")

correlation = df.stat.corr("age", "balance")

print("Correlation:", correlation)

# ======================================================
# 10. LOAN DEFAULT ANALYSIS
# ======================================================

print("\nDEFAULT COUNTS")

default_counts = df.groupBy("default").count()

default_counts.show()

# VISUALIZATION

default_pd = default_counts.toPandas()

plt.figure(figsize=(5, 5))
plt.bar(default_pd["default"], default_pd["count"])
plt.title("Loan Default Counts")
plt.xlabel("Default")
plt.ylabel("Count")

plt.show()

# ======================================================
# 11. CONTACT METHOD ANALYSIS
# ======================================================

print("\nCONTACT METHOD SUCCESS RATE")

contact_analysis = df.groupBy("contact").agg(
    (
        sum(when(col("y") == "yes", 1).otherwise(0))
        / count("*") * 100
    ).alias("success_rate")
)

contact_analysis.show()

# ======================================================
# 12. SPARK SQL
# ======================================================

print("\nSPARK SQL ANALYSIS")

df.createOrReplaceTempView("bank")

spark.sql("""
SELECT
    age_group,
    AVG(balance) AS avg_balance
FROM bank
GROUP BY age_group
""").show()

spark.sql("""
SELECT
    job,
    COUNT(*) AS total_clients
FROM bank
GROUP BY job
ORDER BY total_clients DESC
""").show()

# STOP SPARK SESSION
spark.stop()