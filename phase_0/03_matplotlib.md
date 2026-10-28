# Matplotlib — Visualization for Machine Learning

Matplotlib is the base plotting library in Python. You use it to **see your data** before modeling and to **diagnose your model** after training. If you can't see the data, you're flying blind.

---

## 1. The Basics — How Plotting Works

Every plot in Matplotlib has two key objects:
- **Figure** → the entire window/canvas
- **Axes** → the actual plot area where data is drawn

```python
import matplotlib.pyplot as plt
import numpy as np

# Simplest possible plot
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y)
plt.show()
# Output: A straight line going from (1,2) to (5,10)
```

### Adding labels, title, and making it look good
```python
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y, color='blue', marker='o', linestyle='--', linewidth=2)
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Study Hours vs Exam Score')
plt.grid(True)        # add grid lines
plt.show()
# Output: A dashed blue line with circle markers, labeled axes, title, and grid
```

---

## 2. Line Plot — Trends Over Time

**When to use**: Showing how something changes over time (loss curves, accuracy over epochs, stock prices).

```python
import matplotlib.pyplot as plt
import numpy as np

epochs = np.arange(1, 11)
train_loss = [0.9, 0.7, 0.5, 0.4, 0.35, 0.3, 0.28, 0.25, 0.23, 0.22]
val_loss =   [0.95, 0.75, 0.6, 0.55, 0.53, 0.52, 0.52, 0.53, 0.55, 0.58]

plt.plot(epochs, train_loss, label='Train Loss', color='blue', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', color='red', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()           # shows the label names
plt.grid(True)
plt.show()
# Output: Two lines — blue (train) goes down, red (val) goes down then up
# This pattern shows OVERFITTING — train improves but validation gets worse
```

---

## 3. Scatter Plot — Relationship Between Two Variables

**When to use**: Seeing if two features are related, spotting outliers, visualizing clusters.

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
height = np.random.normal(170, 10, 100)          # 100 heights around 170cm
weight = height * 0.5 + np.random.normal(0, 5, 100)   # weight related to height + noise

plt.scatter(height, weight, color='green', alpha=0.6, edgecolors='black')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Height vs Weight')
plt.show()
# Output: Cloud of green dots trending upward — taller people tend to weigh more
```

### Scatter plot with different colors for classes (classification data)
```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
# Class 0 — centered around (2, 2)
x0 = np.random.randn(50) + 2
y0 = np.random.randn(50) + 2

# Class 1 — centered around (6, 6)
x1 = np.random.randn(50) + 6
y1 = np.random.randn(50) + 6

plt.scatter(x0, y0, color='blue', label='Class 0', alpha=0.7)
plt.scatter(x1, y1, color='red', label='Class 1', alpha=0.7)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Two Classes in Feature Space')
plt.legend()
plt.show()
# Output: Two separate clusters — blue bottom-left, red top-right
```

---

## 4. Histogram — Distribution of a Single Feature

**When to use**: Understanding how a feature is spread out. Is it bell-shaped? Skewed? Has outliers?

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
ages = np.random.normal(35, 10, 1000)    # 1000 ages, average=35, spread=10

plt.hist(ages, bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Count')
plt.title('Distribution of Ages')
plt.axvline(x=np.mean(ages), color='red', linestyle='--', label=f'Mean: {np.mean(ages):.1f}')
plt.legend()
plt.show()
# Output: Bell-shaped curve centered around 35, with a red dashed line at the mean
```

### Comparing distributions of two groups
```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
group_a = np.random.normal(50, 10, 500)
group_b = np.random.normal(60, 15, 500)

plt.hist(group_a, bins=30, alpha=0.5, label='Group A', color='blue')
plt.hist(group_b, bins=30, alpha=0.5, label='Group B', color='orange')
plt.xlabel('Score')
plt.ylabel('Count')
plt.title('Score Distribution: Group A vs Group B')
plt.legend()
plt.show()
# Output: Two overlapping bell curves — Group B shifted right (higher scores)
```

---

## 5. Bar Plot — Comparing Categories

**When to use**: Showing counts or averages across different categories.

```python
import matplotlib.pyplot as plt

departments = ['HR', 'Tech', 'Finance', 'Marketing']
avg_salary = [52000, 75000, 60000, 55000]

plt.bar(departments, avg_salary, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
plt.xlabel('Department')
plt.ylabel('Average Salary')
plt.title('Average Salary by Department')
plt.show()
# Output: 4 colored bars — Tech is the tallest
```

### Horizontal bar (useful when category names are long)
```python
import matplotlib.pyplot as plt

features = ['Income', 'Age', 'Credit Score', 'Loan Amount', 'Employment Years']
importance = [0.35, 0.25, 0.20, 0.12, 0.08]

plt.barh(features, importance, color='teal')
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.show()
# Output: Horizontal bars — Income is the most important feature
```

---

## 6. Subplots — Multiple Plots in One Figure

**When to use**: Comparing multiple plots side by side (very common in EDA and model diagnostics).

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
data1 = np.random.normal(0, 1, 1000)
data2 = np.random.exponential(2, 1000)

# Create 1 row, 2 columns of plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1 (left)
axes[0].hist(data1, bins=30, color='skyblue', edgecolor='black')
axes[0].set_title('Normal Distribution')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Count')

# Plot 2 (right)
axes[1].hist(data2, bins=30, color='salmon', edgecolor='black')
axes[1].set_title('Exponential Distribution')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Count')

plt.tight_layout()     # prevents overlapping
plt.show()
# Output: Two histograms side by side — left is bell-shaped, right is right-skewed
```

### 2x2 grid of plots
```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Top-left: Line plot
x = np.linspace(0, 10, 100)
axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('Sin Wave')

# Top-right: Scatter
axes[0, 1].scatter(np.random.randn(100), np.random.randn(100), alpha=0.5)
axes[0, 1].set_title('Random Scatter')

# Bottom-left: Histogram
axes[1, 0].hist(np.random.randn(500), bins=25, color='green', edgecolor='black')
axes[1, 0].set_title('Normal Distribution')

# Bottom-right: Bar
axes[1, 1].bar(['A', 'B', 'C'], [30, 50, 20], color='purple')
axes[1, 1].set_title('Category Counts')

plt.tight_layout()
plt.show()
# Output: 2x2 grid with 4 different plot types
```

---

## 7. Box Plot — Spotting Outliers and Spread

**When to use**: Seeing the median, spread, and outliers of a feature. Common in data exploration.

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
data = np.random.normal(50, 10, 200)
data = np.append(data, [100, 110, 5])   # add some outliers

plt.boxplot(data, vert=True)
plt.ylabel('Value')
plt.title('Box Plot — Detecting Outliers')
plt.show()
# Output: Box in the middle (25th to 75th percentile), 
#         line inside = median,
#         whiskers extend to min/max (within 1.5x IQR),
#         dots beyond whiskers = OUTLIERS (100, 110, 5)
```

### Comparing groups with box plots
```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
group_a = np.random.normal(60, 10, 100)
group_b = np.random.normal(75, 15, 100)
group_c = np.random.normal(50, 5, 100)

plt.boxplot([group_a, group_b, group_c], labels=['Group A', 'Group B', 'Group C'])
plt.ylabel('Score')
plt.title('Score Distribution by Group')
plt.show()
# Output: 3 box plots side by side — Group B highest and most spread out
```

---

## 8. Heatmap (Manual) — Correlation Matrix

**When to use**: Visualizing how features are correlated with each other. Always do this before modeling.

```python
import matplotlib.pyplot as plt
import numpy as np

# Fake correlation matrix
labels = ['Age', 'Income', 'Score', 'Debt']
corr_matrix = np.array([
    [1.0,  0.6,  0.3, -0.2],
    [0.6,  1.0,  0.8, -0.5],
    [0.3,  0.8,  1.0, -0.4],
    [-0.2, -0.5, -0.4,  1.0]
])

plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Correlation')
plt.xticks(range(4), labels, rotation=45)
plt.yticks(range(4), labels)
plt.title('Feature Correlation Matrix')

# Add numbers on each cell
for i in range(4):
    for j in range(4):
        plt.text(j, i, f'{corr_matrix[i, j]:.1f}', ha='center', va='center', fontsize=12)

plt.tight_layout()
plt.show()
# Output: Color grid — red = positive correlation, blue = negative correlation
# Income and Score are highly correlated (0.8) — might cause issues in linear models
```

---

## 9. Pie Chart — Proportion of Categories

**When to use**: Showing class distribution of your target variable.

```python
import matplotlib.pyplot as plt

labels = ['Not Fraud', 'Fraud']
sizes = [9900, 100]
colors = ['#2ecc71', '#e74c3c']

plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Fraud Detection — Class Distribution')
plt.show()
# Output: Almost entirely green (99%) with a tiny red slice (1%)
# This tells you the dataset is HEAVILY IMBALANCED
```

---

## 10. Saving Plots

```python
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [4, 5, 6])
plt.title('My Plot')

# Save as PNG (high quality)
plt.savefig('my_plot.png', dpi=150, bbox_inches='tight')

# Save as PDF (for papers/reports)
plt.savefig('my_plot.pdf', bbox_inches='tight')

plt.show()
```

> `dpi=150` → higher number = higher resolution
> `bbox_inches='tight'` → removes extra white space around the plot

---

## 11. Common ML Diagnostic Plots

These are plots you'll build after training models:

### Residual Plot (for Regression)
```python
import matplotlib.pyplot as plt
import numpy as np

# Simulated predictions and actual values
np.random.seed(42)
y_actual = np.random.uniform(10, 100, 50)
y_predicted = y_actual + np.random.normal(0, 10, 50)  # predictions with some error
residuals = y_actual - y_predicted

plt.scatter(y_predicted, residuals, alpha=0.6, edgecolors='black')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot')
plt.show()
# Output: Scatter of points around the red zero line
# GOOD: randomly scattered around zero (no pattern)
# BAD: funnel shape or curve (means model is missing something)
```

### Actual vs Predicted (for Regression)
```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
y_actual = np.random.uniform(10, 100, 50)
y_predicted = y_actual + np.random.normal(0, 8, 50)

plt.scatter(y_actual, y_predicted, alpha=0.6, edgecolors='black')
plt.plot([10, 100], [10, 100], color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()
# Output: Points clustered around the red diagonal line
# Closer to the line = better predictions
```

---

## Quick Reference Cheat Sheet

| Plot Type | When to Use | Code |
|---|---|---|
| Line | Trends over time, loss curves | `plt.plot(x, y)` |
| Scatter | Relationships, clusters | `plt.scatter(x, y)` |
| Histogram | Distribution of one feature | `plt.hist(data, bins=30)` |
| Bar | Comparing categories | `plt.bar(categories, values)` |
| Box | Outliers, spread | `plt.boxplot(data)` |
| Heatmap | Correlation matrix | `plt.imshow(matrix, cmap='coolwarm')` |
| Pie | Class proportions | `plt.pie(sizes, labels=labels)` |
| Subplots | Multiple plots | `fig, axes = plt.subplots(rows, cols)` |

| Decoration | Code |
|---|---|
| X/Y label | `plt.xlabel('text')`, `plt.ylabel('text')` |
| Title | `plt.title('text')` |
| Legend | `plt.legend()` (need `label=` in plot) |
| Grid | `plt.grid(True)` |
| Horizontal line | `plt.axhline(y=val, color='red')` |
| Vertical line | `plt.axvline(x=val, color='red')` |
| Save | `plt.savefig('name.png', dpi=150)` |
| Figure size | `plt.figure(figsize=(10, 6))` |

