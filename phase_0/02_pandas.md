# Pandas — Everything You Need for Machine Learning

Pandas is how you load, explore, clean, and prepare real-world data before feeding it to any ML model. Think of it as **Excel on steroids, but in Python**.

The two main objects:
- **Series** → a single column (like one column of a spreadsheet)
- **DataFrame** → a full table (like the entire spreadsheet)

---

## 1. Creating DataFrames

### From a dictionary (most common for quick examples)
```python
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age': [25, 30, 35, 28],
    'Salary': [50000, 60000, 70000, 55000],
    'Department': ['HR', 'Tech', 'Tech', 'HR']
}

df = pd.DataFrame(data)
print(df)
# Output:
#       Name  Age  Salary Department
# 0    Alice   25   50000         HR
# 1      Bob   30   60000       Tech
# 2  Charlie   35   70000       Tech
# 3    Diana   28   55000         HR
```

### From a CSV file (how you'll load real datasets)
```python
# Reading
df = pd.read_csv('data.csv')

# Writing
df.to_csv('output.csv', index=False)   # index=False so it doesn't save row numbers as a column
```

---

## 2. First Look at Your Data

These are the FIRST things you run on any new dataset:

```python
import pandas as pd

df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'Age': [25, 30, 35, 28, None],
    'Salary': [50000, 60000, 70000, 55000, 45000],
    'Department': ['HR', 'Tech', 'Tech', 'HR', 'Finance']
})

# First 5 rows (default) — quick peek
print(df.head())
# Output:
#       Name   Age  Salary Department
# 0    Alice  25.0   50000         HR
# 1      Bob  30.0   60000       Tech
# 2  Charlie  35.0   70000       Tech
# 3    Diana  28.0   55000         HR
# 4      Eve   NaN   45000    Finance

# Last 3 rows
print(df.tail(3))

# Shape — (rows, columns)
print(df.shape)
# Output: (5, 4)

# Column names
print(df.columns)
# Output: Index(['Name', 'Age', 'Salary', 'Department'], dtype='object')

# Data types of each column
print(df.dtypes)
# Output:
# Name           object     ← "object" means text/string
# Age           float64     ← notice it's float because of the NaN
# Salary          int64
# Department     object

# Quick summary — shows data types, non-null counts, memory usage
print(df.info())
# Output:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 5 entries, 0 to 4
# Data columns (total 4 columns):
#  #   Column      Non-Null Count  Dtype
# ---  ------      --------------  -----
#  0   Name        5 non-null      object
#  1   Age         4 non-null      float64    ← only 4 non-null! 1 missing
#  2   Salary      5 non-null      int64
#  3   Department  5 non-null      object

# Statistical summary of numeric columns
print(df.describe())
# Output:
#              Age        Salary
# count   4.000000      5.000000
# mean   29.500000  56000.000000
# std     4.203173   9617.692031
# min    25.000000  45000.000000
# 25%    27.250000  50000.000000
# 50%    29.000000  55000.000000
# 75%    31.250000  60000.000000
# max    35.000000  70000.000000
```

---

## 3. Selecting Data

### Select columns
```python
# Single column → returns a Series
print(df['Name'])
# Output:
# 0      Alice
# 1        Bob
# 2    Charlie
# 3      Diana
# 4        Eve
# Name: Name, dtype: object

# Multiple columns → returns a DataFrame
print(df[['Name', 'Salary']])
# Output:
#       Name  Salary
# 0    Alice   50000
# 1      Bob   60000
# 2  Charlie   70000
# 3    Diana   55000
# 4      Eve   45000
```

### Select rows — loc (by label/name) and iloc (by position/number)
```python
# iloc — integer-based location (like array indexing)
print(df.iloc[0])         # first row
# Output:
# Name          Alice
# Age            25.0
# Salary        50000
# Department       HR

print(df.iloc[0:3])       # first 3 rows
print(df.iloc[0, 2])      # first row, third column → 50000

# loc — label-based (uses column names)
print(df.loc[0, 'Name'])  # → 'Alice'
print(df.loc[0:2, ['Name', 'Age']])   # rows 0 to 2, only Name and Age columns
# Output:
#       Name   Age
# 0    Alice  25.0
# 1      Bob  30.0
# 2  Charlie  35.0
```

> **When to use which?**
> - `iloc` → when you know the position number (row 0, column 2)
> - `loc` → when you know the label/name (column 'Salary')

---

## 4. Filtering Rows (WHERE in SQL)

This is how you select only rows that match a condition.

```python
# People older than 28
print(df[df['Age'] > 28])
# Output:
#       Name   Age  Salary Department
# 1      Bob  30.0   60000       Tech
# 2  Charlie  35.0   70000       Tech

# People in Tech department
print(df[df['Department'] == 'Tech'])
# Output:
#       Name   Age  Salary Department
# 1      Bob  30.0   60000       Tech
# 2  Charlie  35.0   70000       Tech

# Multiple conditions: use & (and), | (or), ~ (not)
# Wrap each condition in parentheses!
print(df[(df['Age'] > 25) & (df['Salary'] > 50000)])
# Output:
#       Name   Age  Salary Department
# 1      Bob  30.0   60000       Tech
# 2  Charlie  35.0   70000       Tech
# 3    Diana  28.0   55000         HR

# Check if value is in a list
print(df[df['Department'].isin(['HR', 'Finance'])])
# Output:
#     Name   Age  Salary Department
# 0  Alice  25.0   50000         HR
# 3  Diana  28.0   55000         HR
# 4    Eve   NaN   45000    Finance
```

---

## 5. Handling Missing Values

Missing data is one of the biggest problems in real ML datasets. Pandas shows missing values as `NaN`.

### Detecting missing values
```python
# Check which cells are missing
print(df.isnull())
# Output:
#     Name    Age  Salary  Department
# 0  False  False   False       False
# 1  False  False   False       False
# 2  False  False   False       False
# 3  False  False   False       False
# 4  False   True   False       False

# Count missing values per column — RUN THIS ON EVERY DATASET
print(df.isnull().sum())
# Output:
# Name          0
# Age           1
# Salary        0
# Department    0

# Percentage missing
print((df.isnull().sum() / len(df)) * 100)
# Output:
# Name           0.0
# Age           20.0
# Salary         0.0
# Department     0.0
```

### Dealing with missing values

```python
# Option 1: Drop rows with any missing values
df_dropped = df.dropna()
print(df_dropped.shape)
# Output: (4, 4)    ← lost 1 row

# Option 2: Fill missing values with a specific value
df_filled = df.fillna({'Age': df['Age'].mean()})   # fill Age NaN with average age
print(df_filled)
# Output:
#       Name    Age  Salary Department
# 0    Alice  25.00   50000         HR
# 1      Bob  30.00   60000       Tech
# 2  Charlie  35.00   70000       Tech
# 3    Diana  28.00   55000         HR
# 4      Eve  29.50   45000    Finance    ← NaN replaced with mean (29.5)

# Option 3: Fill with median (better when data has outliers)
df['Age'] = df['Age'].fillna(df['Age'].median())

# Option 4: Forward fill (use previous row's value) — good for time series
# df['Age'] = df['Age'].ffill()
```

> **Rule of thumb for ML**:
> - Few missing values (<5%) → fill with mean/median
> - Many missing values (>30%) → consider dropping that column
> - Categorical missing → fill with mode (most frequent) or a new category "Unknown"

---

## 6. Value Counts — Understand Your Categories

```python
print(df['Department'].value_counts())
# Output:
# HR         2
# Tech       2
# Finance    1
# Name: Department, dtype: int64

# As percentages
print(df['Department'].value_counts(normalize=True))
# Output:
# HR         0.4
# Tech       0.4
# Finance    0.2
```

**In ML**: Use `value_counts()` on your **target variable** to check for **class imbalance**.
```python
# Example: checking if fraud detection dataset is balanced
# labels.value_counts()
# 0    9900    ← 99% not fraud
# 1     100    ← 1% fraud     ← IMBALANCED!
```

---

## 7. Adding and Modifying Columns

```python
# Add a new column
df['Bonus'] = df['Salary'] * 0.10
print(df[['Name', 'Salary', 'Bonus']])
# Output:
#       Name  Salary   Bonus
# 0    Alice   50000  5000.0
# 1      Bob   60000  6000.0
# 2  Charlie   70000  7000.0
# 3    Diana   55000  5500.0
# 4      Eve   45000  4500.0

# Modify existing column
df['Salary'] = df['Salary'] * 1.1   # 10% raise for everyone

# Create column based on condition
df['Senior'] = df['Age'].apply(lambda x: 'Yes' if x > 30 else 'No')
print(df[['Name', 'Age', 'Senior']])
# Output:
#       Name   Age Senior
# 0    Alice  25.0     No
# 1      Bob  30.0     No
# 2  Charlie  35.0    Yes
# 3    Diana  28.0     No
# 4      Eve  29.5     No

# Drop a column
df = df.drop('Bonus', axis=1)    # axis=1 means column
```

---

## 8. Apply — Run Any Custom Function on Your Data

```python
# Apply a function to a column
df['Name_Length'] = df['Name'].apply(len)
print(df[['Name', 'Name_Length']])
# Output:
#       Name  Name_Length
# 0    Alice           5
# 1      Bob           3
# 2  Charlie           7
# 3    Diana           5
# 4      Eve           3

# Apply a custom function
def categorize_salary(salary):
    if salary > 60000:
        return 'High'
    elif salary > 50000:
        return 'Medium'
    else:
        return 'Low'

df['Salary_Level'] = df['Salary'].apply(categorize_salary)
```

---

## 9. GroupBy — Split, Apply, Combine

GroupBy lets you group rows by a column, then calculate something for each group. Think of it as "for each department, calculate the average salary."

```python
df = pd.DataFrame({
    'Department': ['HR', 'Tech', 'Tech', 'HR', 'Finance'],
    'Salary': [50000, 60000, 70000, 55000, 45000],
    'Age': [25, 30, 35, 28, 22]
})

# Average salary per department
print(df.groupby('Department')['Salary'].mean())
# Output:
# Department
# Finance    45000.0
# HR         52500.0
# Tech       65000.0

# Multiple aggregations at once
print(df.groupby('Department').agg({
    'Salary': ['mean', 'max', 'min'],
    'Age': 'mean'
}))
# Output:
#             Salary                   Age
#               mean    max    min    mean
# Department
# Finance      45000  45000  45000  22.00
# HR           52500  55000  50000  26.50
# Tech         65000  70000  60000  32.50

# Count per group
print(df.groupby('Department').size())
# Output:
# Department
# Finance    1
# HR         2
# Tech       2
```

---

## 10. Sorting

```python
# Sort by one column
print(df.sort_values('Salary', ascending=False))
# Output:
#   Department  Salary  Age
# 2       Tech   70000   35
# 1       Tech   60000   30
# 3         HR   55000   28
# 0         HR   50000   25
# 4    Finance   45000   22

# Sort by multiple columns
print(df.sort_values(['Department', 'Salary'], ascending=[True, False]))
```

---

## 11. Merging / Joining DataFrames

When your data is split across multiple files/tables.

```python
# Employee info
employees = pd.DataFrame({
    'EmpID': [1, 2, 3, 4],
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'DeptID': [101, 102, 102, 101]
})

# Department info
departments = pd.DataFrame({
    'DeptID': [101, 102, 103],
    'DeptName': ['HR', 'Tech', 'Finance']
})

# Merge (like SQL JOIN)
merged = pd.merge(employees, departments, on='DeptID', how='left')
print(merged)
# Output:
#    EmpID     Name  DeptID DeptName
# 0      1    Alice     101       HR
# 1      2      Bob     102     Tech
# 2      3  Charlie     102     Tech
# 3      4    Diana     101       HR
```

> **Join types**:
> - `how='inner'` → only matching rows (default)
> - `how='left'` → all rows from left table, matching from right
> - `how='right'` → all rows from right table
> - `how='outer'` → all rows from both

---

## 12. Encoding Categorical Variables (CRITICAL for ML)

ML models only understand numbers. You need to convert text categories to numbers.

### One-Hot Encoding (for nominal categories — no order)
```python
df = pd.DataFrame({
    'Color': ['Red', 'Blue', 'Green', 'Blue', 'Red'],
    'Size': [10, 20, 15, 25, 12]
})

# get_dummies creates a new column for each category
encoded = pd.get_dummies(df, columns=['Color'], dtype=int)
print(encoded)
# Output:
#    Size  Color_Blue  Color_Green  Color_Red
# 0    10           0            0          1
# 1    20           1            0          0
# 2    15           0            1          0
# 3    25           1            0          0
# 4    12           0            0          1

# Drop first to avoid multicollinearity (dummy variable trap)
encoded = pd.get_dummies(df, columns=['Color'], drop_first=True, dtype=int)
print(encoded)
# Output:
#    Size  Color_Green  Color_Red
# 0    10            0          1
# 1    20            0          0
# 2    15            1          0
# 3    25            0          0
# 4    12            0          1
```

### Label Encoding (for ordinal categories — has order)
```python
# When categories have a natural order
size_map = {'Small': 0, 'Medium': 1, 'Large': 2}
df = pd.DataFrame({'Size': ['Medium', 'Small', 'Large', 'Medium']})
df['Size_Encoded'] = df['Size'].map(size_map)
print(df)
# Output:
#      Size  Size_Encoded
# 0  Medium             1
# 1   Small             0
# 2   Large             2
# 3  Medium             1
```

---

## 13. Correlation — See Which Features Are Related

```python
df = pd.DataFrame({
    'Hours_Studied': [1, 2, 3, 4, 5],
    'Exam_Score': [40, 50, 60, 70, 80],
    'Sleep_Hours': [8, 7, 7, 6, 5]
})

print(df.corr())
# Output:
#                 Hours_Studied  Exam_Score  Sleep_Hours
# Hours_Studied        1.000000    1.000000    -0.948683
# Exam_Score           1.000000    1.000000    -0.948683
# Sleep_Hours         -0.948683   -0.948683     1.000000

# Hours_Studied and Exam_Score have correlation = 1.0 (perfect positive)
# Sleep_Hours has -0.95 correlation with both (negative — more study, less sleep)
```

> **Correlation values**:
> - Close to **+1**: as one goes up, the other goes up
> - Close to **-1**: as one goes up, the other goes down
> - Close to **0**: no linear relationship

---

## 14. Useful String Operations

```python
df = pd.DataFrame({'Name': ['  Alice  ', 'BOB', 'charlie', 'Diana M.']})

df['Clean'] = df['Name'].str.strip()         # remove extra spaces
df['Lower'] = df['Name'].str.lower()         # lowercase
df['Upper'] = df['Name'].str.upper()         # uppercase
df['Contains_a'] = df['Name'].str.contains('a', case=False)  # search
df['Length'] = df['Name'].str.len()

print(df)
# Output:
#         Name    Clean    Lower     Upper  Contains_a  Length
# 0    Alice    Alice    alice      ALICE        True       9
# 1        BOB      BOB      bob        BOB       False       3
# 2    charlie  charlie  charlie    CHARLIE        True       7
# 3   Diana M. Diana M. diana m.  DIANA M.        True       8
```

---

## 15. Handling Duplicates

```python
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Alice', 'Charlie'],
    'Age': [25, 30, 25, 35]
})

# Check for duplicates
print(df.duplicated())
# Output:
# 0    False
# 1    False
# 2     True    ← duplicate of row 0
# 3    False

# Count duplicates
print(df.duplicated().sum())
# Output: 1

# Remove duplicates
df_clean = df.drop_duplicates()
print(df_clean)
# Output:
#       Name  Age
# 0    Alice   25
# 1      Bob   30
# 3  Charlie   35
```

---

## 16. Pivot Table — Summarize Data Like Excel

```python
df = pd.DataFrame({
    'Department': ['HR', 'Tech', 'HR', 'Tech', 'HR'],
    'Gender': ['M', 'F', 'F', 'M', 'M'],
    'Salary': [50000, 60000, 55000, 70000, 52000]
})

pivot = pd.pivot_table(df, values='Salary', index='Department',
                       columns='Gender', aggfunc='mean')
print(pivot)
# Output:
# Gender          F        M
# Department
# HR          55000.0  51000.0
# Tech        60000.0  70000.0
```

---

## 17. Practical ML Data Prep Pipeline (Putting It All Together)

```python
import pandas as pd
import numpy as np

# Step 1: Load data
# df = pd.read_csv('dataset.csv')

# Step 2: Quick look
# df.head()
# df.shape
# df.info()
# df.describe()

# Step 3: Check missing values
# df.isnull().sum()

# Step 4: Handle missing values
# df['numeric_col'] = df['numeric_col'].fillna(df['numeric_col'].median())
# df['category_col'] = df['category_col'].fillna('Unknown')

# Step 5: Remove duplicates
# df = df.drop_duplicates()

# Step 6: Check target variable distribution
# df['target'].value_counts()

# Step 7: Encode categorical variables
# df = pd.get_dummies(df, columns=['cat_col1', 'cat_col2'], drop_first=True)

# Step 8: Check correlations
# df.corr()

# Step 9: Separate features (X) and target (y)
# X = df.drop('target', axis=1)
# y = df['target']
```

---

## Quick Reference Cheat Sheet

| What you want to do | Code |
|---|---|
| Load CSV | `pd.read_csv('file.csv')` |
| First look | `df.head()`, `df.shape`, `df.info()`, `df.describe()` |
| Data types | `df.dtypes` |
| Select column | `df['col']` or `df[['col1', 'col2']]` |
| Select rows by position | `df.iloc[0:5]` |
| Select rows by label | `df.loc[0:5, 'col']` |
| Filter rows | `df[df['col'] > value]` |
| Missing values count | `df.isnull().sum()` |
| Fill missing | `df['col'].fillna(value)` |
| Drop missing | `df.dropna()` |
| Value counts | `df['col'].value_counts()` |
| Add column | `df['new'] = expression` |
| Apply function | `df['col'].apply(func)` |
| GroupBy | `df.groupby('col')['val'].mean()` |
| Sort | `df.sort_values('col', ascending=False)` |
| Merge | `pd.merge(df1, df2, on='key')` |
| One-hot encode | `pd.get_dummies(df, columns=['col'])` |
| Correlation | `df.corr()` |
| Duplicates | `df.duplicated().sum()`, `df.drop_duplicates()` |
| Drop column | `df.drop('col', axis=1)` |
| Save CSV | `df.to_csv('out.csv', index=False)` |

