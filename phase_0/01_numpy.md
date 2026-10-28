# NumPy — Everything You Need for Machine Learning

NumPy is the backbone of ML in Python. Every dataset, every calculation, every model internally uses NumPy arrays. Think of it as **supercharged lists that can do math really fast**.

---

## 1. Why NumPy Instead of Normal Python Lists?

Python lists are slow because each element is stored separately in memory. NumPy stores everything together in a block, so operations run **50-100x faster**.

```python
import numpy as np

# Python list way (slow)
python_list = [1, 2, 3, 4, 5]
doubled = [x * 2 for x in python_list]
print(doubled)
# Output: [2, 4, 6, 8, 10]

# NumPy way (fast, clean)
arr = np.array([1, 2, 3, 4, 5])
doubled = arr * 2
print(doubled)
# Output: [ 2  4  6  8 10]
```

**Key takeaway**: In NumPy, operations apply to the **entire array at once** — no loops needed. This is called **vectorization**.

---

## 2. Creating Arrays

### From a list
```python
import numpy as np

a = np.array([1, 2, 3])            # 1D array (like a row)
print(a)
# Output: [1 2 3]

b = np.array([[1, 2, 3],
              [4, 5, 6]])           # 2D array (like a table)
print(b)
# Output:
# [[1 2 3]
#  [4 5 6]]
```

### Ready-made arrays (used all the time in ML)
```python
# All zeros — used to initialize weights or create empty containers
zeros = np.zeros((3, 4))       # 3 rows, 4 columns
print(zeros)
# Output:
# [[0. 0. 0. 0.]
#  [0. 0. 0. 0.]
#  [0. 0. 0. 0.]]

# All ones
ones = np.ones((2, 3))
print(ones)
# Output:
# [[1. 1. 1.]
#  [1. 1. 1.]]

# Identity matrix — diagonal is 1, rest 0 (used in linear algebra, Ridge regression)
eye = np.eye(3)
print(eye)
# Output:
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]]

# Evenly spaced numbers — useful for plotting graphs
x = np.linspace(0, 10, 5)     # 5 numbers from 0 to 10
print(x)
# Output: [ 0.   2.5  5.   7.5 10. ]

# Range of numbers
r = np.arange(0, 10, 2)       # start=0, stop=10(not included), step=2
print(r)
# Output: [0 2 4 6 8]
```

### Random arrays (used for random initialization, shuffling, sampling)
```python
# Random numbers between 0 and 1
rand = np.random.rand(3, 3)
print(rand)
# Output (will vary):
# [[0.42 0.71 0.12]
#  [0.53 0.84 0.29]
#  [0.67 0.43 0.91]]

# Random integers
rand_int = np.random.randint(1, 100, size=(2, 4))  # random ints from 1 to 99
print(rand_int)
# Output (will vary):
# [[45 82  3 67]
#  [12 91 55 28]]

# Random from normal distribution (bell curve) — mean=0, std=1
normal = np.random.randn(5)
print(normal)
# Output (will vary): [-0.52  1.23 -0.15  0.87 -1.41]

# Set seed for reproducibility (same random numbers every time)
np.random.seed(42)
print(np.random.rand(3))
# Output: [0.37454012 0.95071431 0.73199394]
```

---

## 3. Array Shape and Reshaping

**Shape** tells you the size of your data. In ML, you'll constantly check and reshape data.

```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

print(arr.shape)      # (rows, columns)
# Output: (2, 3)

print(arr.ndim)       # number of dimensions
# Output: 2

print(arr.size)       # total number of elements
# Output: 6
```

### Reshape — changing the arrangement without changing data
```python
a = np.array([1, 2, 3, 4, 5, 6])
print(a.shape)
# Output: (6,)

# Reshape to 2 rows, 3 columns
b = a.reshape(2, 3)
print(b)
# Output:
# [[1 2 3]
#  [4 5 6]]

# Reshape to 3 rows, 2 columns
c = a.reshape(3, 2)
print(c)
# Output:
# [[1 2]
#  [3 4]
#  [5 6]]

# Use -1 to let NumPy figure out one dimension automatically
d = a.reshape(-1, 1)   # make it a column vector (very common in ML)
print(d)
# Output:
# [[1]
#  [2]
#  [3]
#  [4]
#  [5]
#  [6]]
print(d.shape)
# Output: (6, 1)
```

### Flatten — make any array back to 1D
```python
arr = np.array([[1, 2], [3, 4], [5, 6]])
flat = arr.flatten()
print(flat)
# Output: [1 2 3 4 5 6]
```

### Transpose — swap rows and columns
```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
print(arr.shape)      # (2, 3)

t = arr.T
print(t)
# Output:
# [[1 4]
#  [2 5]
#  [3 6]]
print(t.shape)        # (3, 2)
```

---

## 4. Indexing and Slicing

### Basic indexing (works like Python lists)
```python
a = np.array([10, 20, 30, 40, 50])

print(a[0])     # first element
# Output: 10

print(a[-1])    # last element
# Output: 50

print(a[1:4])   # elements at index 1, 2, 3
# Output: [20 30 40]
```

### 2D indexing — [row, column]
```python
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

print(arr[0, 0])    # first row, first column
# Output: 1

print(arr[1, 2])    # second row, third column
# Output: 6

print(arr[0])       # entire first row
# Output: [1 2 3]

print(arr[:, 1])    # entire second column (all rows, column index 1)
# Output: [2 5 8]

print(arr[:2, :2])  # first 2 rows, first 2 columns
# Output:
# [[1 2]
#  [4 5]]
```

### Boolean indexing — filter based on condition (VERY useful in ML)
```python
arr = np.array([15, 25, 35, 45, 55])

# Get all values greater than 30
mask = arr > 30
print(mask)
# Output: [False False  True  True  True]

print(arr[mask])
# Output: [35 45 55]

# Or directly:
print(arr[arr > 30])
# Output: [35 45 55]

# Multiple conditions (use & for AND, | for OR)
print(arr[(arr > 20) & (arr < 50)])
# Output: [25 35 45]
```

---

## 5. Math Operations (Element-wise)

When you do math with NumPy arrays, it happens **element by element**.

```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

print(a + b)     # Output: [11 22 33 44]
print(a - b)     # Output: [ -9 -18 -27 -36]
print(a * b)     # Output: [ 10  40  90 160]
print(a / b)     # Output: [0.1 0.1 0.1 0.1]
print(a ** 2)    # Output: [ 1  4  9 16]       (each element squared)
print(np.sqrt(a))# Output: [1.   1.41 1.73 2.  ]
print(np.exp(a)) # Output: [ 2.72  7.39 20.09 54.60]  (e^x — used in softmax, sigmoid)
print(np.log(a)) # Output: [0.   0.69 1.10 1.39]      (natural log — used in cross-entropy loss)
```

---

## 6. Aggregation Functions (Summarize Data)

These are your go-to for understanding your dataset at a glance.

```python
arr = np.array([10, 20, 30, 40, 50])

print(np.sum(arr))      # Output: 150
print(np.mean(arr))     # Output: 30.0      (average)
print(np.median(arr))   # Output: 30.0
print(np.std(arr))      # Output: 14.14     (standard deviation — spread of data)
print(np.var(arr))      # Output: 200.0     (variance = std^2)
print(np.min(arr))      # Output: 10
print(np.max(arr))      # Output: 50
print(np.argmin(arr))   # Output: 0         (INDEX of minimum — used in KNN, clustering)
print(np.argmax(arr))   # Output: 4         (INDEX of maximum — used in classification predictions)
```

### Aggregation along axes (rows vs columns)
```python
arr = np.array([[1, 2, 3],
                [4, 5, 6]])

print(np.sum(arr))            # total sum → 21
print(np.sum(arr, axis=0))    # sum along rows (column-wise) → [5 7 9]
print(np.sum(arr, axis=1))    # sum along columns (row-wise) → [ 6 15]
print(np.mean(arr, axis=0))   # mean of each column → [2.5 3.5 4.5]
print(np.mean(arr, axis=1))   # mean of each row → [2. 5.]
```

> **How to remember axis?**
> - `axis=0` → goes **down** (operates across rows, gives one value per column)
> - `axis=1` → goes **across** (operates across columns, gives one value per row)

---

## 7. Broadcasting

Broadcasting is how NumPy handles operations between arrays of **different shapes**. Instead of throwing an error, it "stretches" the smaller array to match.

```python
# Adding a single number to every element
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
print(arr + 10)
# Output:
# [[11 12 13]
#  [14 15 16]]

# Adding a 1D array to each row of a 2D array
row = np.array([100, 200, 300])
print(arr + row)
# Output:
# [[101 202 303]
#  [104 205 306]]
# NumPy automatically added [100, 200, 300] to EACH row

# Mean centering (subtract column mean from each column) — used before PCA
data = np.array([[10, 200, 3000],
                 [20, 400, 6000],
                 [30, 600, 9000]])

col_means = data.mean(axis=0)        # mean of each column
print(col_means)
# Output: [  20.  400. 6000.]

centered = data - col_means           # broadcasting subtracts from each row
print(centered)
# Output:
# [[ -10. -200. -3000.]
#  [   0.    0.     0.]
#  [  10.  200.  3000.]]
```

---

## 8. Dot Product and Matrix Multiplication

This is THE most important operation in ML. Every model prediction is basically a dot product.

### Dot product of 1D arrays (returns a single number)
```python
# Think of it as: multiply matching elements, then sum
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

result = np.dot(a, b)   # (1*4) + (2*5) + (3*6) = 4 + 10 + 18 = 32
print(result)
# Output: 32
```

**Why this matters**: In Linear Regression, prediction = dot product of features and weights.
```
prediction = w1*x1 + w2*x2 + w3*x3 = dot(weights, features)
```

### Matrix multiplication (2D)
```python
# For matrix multiplication: columns of first = rows of second
A = np.array([[1, 2],
              [3, 4]])     # shape (2, 2)

B = np.array([[5, 6],
              [7, 8]])     # shape (2, 2)

result = np.dot(A, B)      # or A @ B  (@ is shorthand)
print(result)
# Output:
# [[19 22]
#  [43 50]]
# How: row1 of A dot col1 of B = 1*5 + 2*7 = 19
#      row1 of A dot col2 of B = 1*6 + 2*8 = 22
#      row2 of A dot col1 of B = 3*5 + 4*7 = 43
#      row2 of A dot col2 of B = 3*6 + 4*8 = 50
```

### Practical ML example: making predictions
```python
# 3 samples, each with 4 features
X = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])    # shape (3, 4)

# 4 weights (one per feature)
weights = np.array([0.1, 0.2, 0.3, 0.4])   # shape (4,)

# Predictions = X @ weights
predictions = X @ weights
print(predictions)
# Output: [ 3. 7. 11.]
# Sample 1: 1*0.1 + 2*0.2 + 3*0.3 + 4*0.4 = 0.1 + 0.4 + 0.9 + 1.6 = 3.0
```

---

## 9. Stacking and Concatenating Arrays

Used when combining datasets, adding new features, or joining arrays.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Vertical stack (add as new rows)
v = np.vstack([a, b])
print(v)
# Output:
# [[1 2 3]
#  [4 5 6]]

# Horizontal stack (add as new columns)
h = np.hstack([a, b])
print(h)
# Output: [1 2 3 4 5 6]

# Column stack — turn 1D arrays into columns of a 2D array
c = np.column_stack([a, b])
print(c)
# Output:
# [[1 4]
#  [2 5]
#  [3 6]]
```

### Concatenate
```python
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6]])

# Add rows (axis=0)
result = np.concatenate([arr1, arr2], axis=0)
print(result)
# Output:
# [[1 2]
#  [3 4]
#  [5 6]]
```

---

## 10. np.where — Conditional Selection

Works like an if-else for every element. Very handy in ML for threshold-based predictions.

```python
arr = np.array([10, 25, 30, 45, 50])

# If value > 30, keep it. Otherwise, replace with 0.
result = np.where(arr > 30, arr, 0)
print(result)
# Output: [ 0  0  0 45 50]

# Convert probabilities to class labels (threshold = 0.5)
probabilities = np.array([0.2, 0.8, 0.4, 0.9, 0.1])
predictions = np.where(probabilities >= 0.5, 1, 0)
print(predictions)
# Output: [0 1 0 1 0]
```

---

## 11. Sorting

```python
arr = np.array([50, 10, 40, 20, 30])

print(np.sort(arr))
# Output: [10 20 30 40 50]

# Get the indices that would sort the array (useful for ranking)
print(np.argsort(arr))
# Output: [1 3 4 2 0]
# Meaning: index 1 (value 10) is smallest, then index 3 (value 20), etc.
```

---

## 12. Unique Values and Counts

```python
labels = np.array([0, 1, 1, 0, 2, 2, 2, 1, 0])

print(np.unique(labels))
# Output: [0 1 2]

values, counts = np.unique(labels, return_counts=True)
print(values)
# Output: [0 1 2]
print(counts)
# Output: [3 3 3]
# Class 0 appears 3 times, class 1 appears 3 times, class 2 appears 3 times
```

---

## 13. Useful Math Functions for ML

```python
# Clip values to a range (prevent extreme values)
arr = np.array([-5, -1, 0, 3, 10])
print(np.clip(arr, 0, 5))
# Output: [0 0 0 3 5]

# Absolute values
print(np.abs(np.array([-3, -1, 0, 2, 4])))
# Output: [3 1 0 2 4]

# Round
print(np.round(np.array([1.234, 5.678, 9.012]), 2))
# Output: [1.23 5.68 9.01]

# Sigmoid function (used in Logistic Regression)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.array([-2, -1, 0, 1, 2])
print(sigmoid(z))
# Output: [0.119 0.269 0.5   0.731 0.881]
# Notice: 0 maps to 0.5, negatives go toward 0, positives go toward 1
```

---

## 14. Copy vs View — A Common Bug Source

```python
arr = np.array([1, 2, 3, 4, 5])

# This creates a VIEW (shared memory — changes affect original!)
view = arr[1:4]
view[0] = 999
print(arr)
# Output: [  1 999   3   4   5]   ← original changed!

# This creates a COPY (independent — safe)
arr = np.array([1, 2, 3, 4, 5])
copy = arr[1:4].copy()
copy[0] = 999
print(arr)
# Output: [1 2 3 4 5]            ← original is safe
```

> **Rule of thumb**: When you slice an array and plan to modify it, use `.copy()` to be safe.

---

## Quick Reference Cheat Sheet

| What you want to do | Code |
|---|---|
| Create array | `np.array([1,2,3])` |
| Zeros/Ones | `np.zeros((r,c))`, `np.ones((r,c))` |
| Random | `np.random.rand(r,c)`, `np.random.randn(r,c)` |
| Shape | `arr.shape` |
| Reshape | `arr.reshape(r,c)` |
| Column vector | `arr.reshape(-1, 1)` |
| Transpose | `arr.T` |
| Select column | `arr[:, col_index]` |
| Filter | `arr[arr > value]` |
| Dot product | `np.dot(a, b)` or `a @ b` |
| Sum/Mean/Std | `np.sum()`, `np.mean()`, `np.std()` |
| Index of max | `np.argmax(arr)` |
| Conditional | `np.where(condition, yes_val, no_val)` |
| Stack | `np.vstack()`, `np.hstack()`, `np.column_stack()` |
| Unique counts | `np.unique(arr, return_counts=True)` |
| Sort | `np.sort(arr)`, `np.argsort(arr)` |
| Clip | `np.clip(arr, min, max)` |
| Safe copy | `arr.copy()` |

