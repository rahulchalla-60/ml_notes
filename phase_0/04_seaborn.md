# Seaborn — Beautiful Statistical Plots for ML

Seaborn is built on top of Matplotlib. It makes prettier plots with **less code** and is designed specifically for **statistical visualization**. Think of it as Matplotlib's smarter, better-looking sibling.

---

## 1. Setup — Seaborn Basics

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set a clean style (do this once at the top of your notebook)
sns.set_style('whitegrid')       # options: 'darkgrid', 'whitegrid', 'dark', 'white', 'ticks'
sns.set_palette('husl')          # nice default color palette
```

### We'll use this sample dataset throughout:
```python
np.random.seed(42)

df = pd.DataFrame({
    'Age': np.random.randint(20, 60, 200),
    'Income': np.random.randint(30000, 120000, 200),
    'SpendingScore': np.random.randint(1, 100, 200),
    'Gender': np.random.choice(['Male', 'Female'], 200),
    'Cluster': np.random.choice([0, 1, 2], 200)
})
```

---

## 2. Histogram / Distribution Plot — How Is a Feature Spread Out?

**When to use**: First thing to check for every numeric feature. Is it normal? Skewed? Has weird gaps?

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
ages = np.random.normal(35, 10, 1000)

sns.histplot(ages, bins=30, kde=True, color='steelblue')
plt.xlabel('Age')
plt.title('Age Distribution')
plt.show()
# Output: Histogram bars with a smooth curve (KDE) overlaid
# KDE = Kernel Density Estimate — a smooth version of the histogram
# Helps you see the overall shape without worrying about bin sizes
```

### KDE plot only (smooth curve without bars)
```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
group_a = np.random.normal(50, 10, 500)
group_b = np.random.normal(65, 12, 500)

sns.kdeplot(group_a, label='Group A', fill=True, alpha=0.3)
sns.kdeplot(group_b, label='Group B', fill=True, alpha=0.3)
plt.xlabel('Score')
plt.title('Score Distribution by Group')
plt.legend()
plt.show()
# Output: Two smooth filled curves — easy to compare distributions
```

---

## 3. Count Plot — Bar Chart for Categorical Data

**When to use**: Checking how many samples you have in each category. Critical for checking **class imbalance** in your target variable.

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    'Gender': np.random.choice(['Male', 'Female'], 200),
    'Cluster': np.random.choice([0, 1, 2], 200)
})

sns.countplot(x='Gender', data=df)
plt.title('Gender Distribution')
plt.show()
# Output: Two bars showing count of Males and Females

# Check class balance of target variable
sns.countplot(x='Cluster', data=df, palette='viridis')
plt.title('Target Variable (Cluster) Distribution')
plt.show()
# Output: 3 bars — if roughly equal, your data is balanced
#          If one bar is much taller/shorter, you have class imbalance
```

---

## 4. Box Plot — Outliers and Spread Across Groups

**When to use**: Comparing a numeric feature across different categories. Instantly shows outliers.

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    'Income': np.concatenate([
        np.random.normal(50000, 10000, 100),
        np.random.normal(80000, 15000, 100)
    ]),
    'Gender': ['Male']*100 + ['Female']*100
})

sns.boxplot(x='Gender', y='Income', data=df, palette='Set2')
plt.title('Income by Gender')
plt.show()
# Output: Two box plots side by side
# The box = middle 50% of data (25th to 75th percentile)
# Line inside box = median
# Whiskers = range of "normal" data
# Dots beyond whiskers = OUTLIERS
```

---

## 5. Violin Plot — Box Plot + Distribution Shape

**When to use**: When you want to see both the spread AND the shape (is it bimodal? skewed?).

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    'Score': np.concatenate([
        np.random.normal(60, 10, 150),
        np.random.normal(80, 5, 150)
    ]),
    'Group': ['A']*150 + ['B']*150
})

sns.violinplot(x='Group', y='Score', data=df, palette='muted')
plt.title('Score Distribution by Group')
plt.show()
# Output: Violin shapes — wider = more data points at that value
# Group A: wider spread (std=10)
# Group B: narrower, concentrated around 80 (std=5)
```

---

## 6. Scatter Plot — Relationship Between Two Numeric Features

**When to use**: Checking if two features are related. Adding `hue` shows how classes separate.

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    'Age': np.random.randint(20, 60, 200),
    'Income': np.random.randint(30000, 120000, 200),
    'Cluster': np.random.choice([0, 1, 2], 200)
})

sns.scatterplot(x='Age', y='Income', hue='Cluster', data=df, palette='deep', alpha=0.7)
plt.title('Age vs Income (colored by Cluster)')
plt.show()
# Output: Scatter plot where each cluster has a different color
# If clusters form visible groups → your features can separate the classes
# If all mixed together → these features alone can't distinguish the classes
```

---

## 7. Heatmap — Correlation Matrix (THE MOST IMPORTANT PLOT)

**When to use**: ALWAYS before building any model. Shows which features are related to each other and to the target.

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    'Hours_Studied': np.random.uniform(1, 10, 100),
    'Sleep_Hours': np.random.uniform(4, 9, 100),
    'Previous_Score': np.random.uniform(30, 90, 100),
})
df['Exam_Score'] = df['Hours_Studied'] * 5 + df['Previous_Score'] * 0.3 + np.random.normal(0, 5, 100)

corr = df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.show()
# Output: Color grid with numbers
# 
# What to look for:
# 1. Features strongly correlated with TARGET (Exam_Score) → good predictors
# 2. Features strongly correlated with EACH OTHER → potential multicollinearity
#    (problematic for linear models)
# 
# Red = positive correlation (both go up together)
# Blue = negative correlation (one goes up, other goes down)
# White/near 0 = no relationship
```

> **annot=True** → shows the actual numbers on the cells
> **fmt='.2f'** → show 2 decimal places
> **cmap='coolwarm'** → red-white-blue color scheme
> **center=0** → white color is at 0 (neutral)

---

## 8. Pair Plot — All Pairwise Relationships at Once

**When to use**: Quick overview of relationships between ALL numeric features. Best for datasets with fewer than 8-10 features (gets too big otherwise).

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    'Feature_1': np.random.randn(150),
    'Feature_2': np.random.randn(150),
    'Feature_3': np.random.randn(150),
    'Class': np.random.choice(['A', 'B', 'C'], 150)
})

sns.pairplot(df, hue='Class', palette='bright', diag_kind='kde')
plt.suptitle('Pair Plot of All Features', y=1.02)
plt.show()
# Output: Grid of scatter plots
# - Diagonal: distribution of each feature (separated by class color)
# - Off-diagonal: scatter plot of every pair of features (colored by class)
# 
# What to look for:
# - Are classes separable in any pair of features? (distinct colored clusters)
# - Any strong linear relationships between features? (diagonal patterns)
```

> **Tip**: For large datasets, use `pairplot(df.sample(500), ...)` to plot only 500 random samples — otherwise it takes forever.

---

## 9. Joint Plot — Two Features in Detail

**When to use**: Deep dive into the relationship between exactly two features — shows both the scatter and the individual distributions.

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)
x = np.random.normal(50, 10, 300)
y = x * 0.8 + np.random.normal(0, 8, 300)

sns.jointplot(x=x, y=y, kind='scatter', color='purple', alpha=0.5)
plt.show()
# Output: Central scatter plot + histogram on top (for x) + histogram on right (for y)
```

### Different kinds of joint plots
```python
# kind='hex' — good for large datasets (like a 2D histogram)
# sns.jointplot(x=x, y=y, kind='hex')

# kind='kde' — smooth contour map (shows density)
# sns.jointplot(x=x, y=y, kind='kde')

# kind='reg' — scatter + regression line
# sns.jointplot(x=x, y=y, kind='reg')
```

---

## 10. Bar Plot (with statistical estimation) — Mean + Confidence Interval

**When to use**: Comparing the average of a numeric feature across categories, with uncertainty.

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    'Department': np.random.choice(['HR', 'Tech', 'Finance'], 200),
    'Salary': np.random.randint(40000, 100000, 200)
})

sns.barplot(x='Department', y='Salary', data=df, palette='pastel', ci=95)
plt.title('Average Salary by Department (with 95% confidence interval)')
plt.show()
# Output: Bars showing mean salary per department
# Black lines on top of each bar = confidence interval
# If confidence intervals overlap a lot → difference might not be significant
```

---

## 11. Swarm Plot — See Every Data Point

**When to use**: When you have small-medium data and want to see every individual point (unlike box/violin which summarize).

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    'Category': np.random.choice(['A', 'B', 'C'], 60),
    'Value': np.random.randn(60) * 10 + 50
})

sns.swarmplot(x='Category', y='Value', data=df, palette='Set1', size=6)
plt.title('Individual Data Points by Category')
plt.show()
# Output: Dots spread out horizontally so they don't overlap
# You can see every single data point — no information is hidden
```

---

## 12. Combining Plots — The Power Move

You can layer Seaborn plots on top of each other for maximum information:

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(42)
df = pd.DataFrame({
    'Group': np.random.choice(['A', 'B'], 100),
    'Score': np.concatenate([
        np.random.normal(60, 10, 50),
        np.random.normal(75, 8, 50)
    ])
})

# Box plot + Strip plot overlay
sns.boxplot(x='Group', y='Score', data=df, palette='Set2', width=0.5)
sns.stripplot(x='Group', y='Score', data=df, color='black', alpha=0.3, size=4)
plt.title('Box Plot with Individual Data Points')
plt.show()
# Output: Box plots showing summary statistics WITH every individual point visible
```

---

## 13. Styling and Customization

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Built-in styles
sns.set_style('whitegrid')     # clean, minimal — best for presentations
# sns.set_style('darkgrid')   # dark background with grid
# sns.set_style('ticks')      # clean with tick marks only

# Built-in color palettes
# sns.set_palette('deep')     # bold colors
# sns.set_palette('muted')    # softer colors
# sns.set_palette('pastel')   # light colors
# sns.set_palette('bright')   # vivid colors
# sns.set_palette('Set2')     # from ColorBrewer
# sns.set_palette('viridis')  # perceptually uniform

# Change font scale (bigger text)
sns.set_context('talk')        # options: 'paper', 'notebook' (default), 'talk', 'poster'
```

---

## 14. Practical EDA Flow with Seaborn (Putting It All Together)

When you get a new dataset, here's the exact sequence of Seaborn plots to make:

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Assuming df is your loaded dataset and 'target' is your label column

# STEP 1: Check target variable balance
# sns.countplot(x='target', data=df)
# plt.title('Target Variable Distribution')
# plt.show()

# STEP 2: Correlation heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
# plt.title('Correlation Matrix')
# plt.show()

# STEP 3: Distribution of each numeric feature
# for col in df.select_dtypes(include='number').columns:
#     sns.histplot(df[col], kde=True)
#     plt.title(f'Distribution of {col}')
#     plt.show()

# STEP 4: Box plots — numeric features vs target (find useful features)
# for col in df.select_dtypes(include='number').columns:
#     if col != 'target':
#         sns.boxplot(x='target', y=col, data=df)
#         plt.title(f'{col} by Target')
#         plt.show()

# STEP 5: Pair plot (if ≤ 8 numeric features)
# sns.pairplot(df, hue='target', diag_kind='kde')
# plt.show()
```

---

## Seaborn vs Matplotlib: When to Use Which?

| Situation | Use |
|---|---|
| Quick statistical plot | **Seaborn** |
| Correlation heatmap | **Seaborn** (`sns.heatmap`) |
| Distribution with KDE | **Seaborn** (`sns.histplot`) |
| Box/Violin plots | **Seaborn** |
| Pair plot | **Seaborn** (Matplotlib can't do this easily) |
| Custom/complex layouts | **Matplotlib** |
| Fine-grained control | **Matplotlib** |
| Loss curves, training plots | **Matplotlib** |
| Saving/exporting | **Matplotlib** (`plt.savefig`) |

> **They work together**: Seaborn creates the plot, and you use Matplotlib's `plt.title()`, `plt.xlabel()`, `plt.savefig()`, etc. to customize and save.

---

## Quick Reference Cheat Sheet

| Plot | Code | When |
|---|---|---|
| Distribution | `sns.histplot(data, kde=True)` | See how a feature is spread |
| KDE only | `sns.kdeplot(data, fill=True)` | Smooth distribution curve |
| Count | `sns.countplot(x='col', data=df)` | Category frequencies |
| Box | `sns.boxplot(x='cat', y='num', data=df)` | Outliers, spread by group |
| Violin | `sns.violinplot(x='cat', y='num', data=df)` | Shape + spread by group |
| Scatter | `sns.scatterplot(x='a', y='b', hue='c', data=df)` | Feature relationships |
| Heatmap | `sns.heatmap(df.corr(), annot=True)` | Correlation matrix |
| Pair plot | `sns.pairplot(df, hue='target')` | All pairwise relationships |
| Joint | `sns.jointplot(x=x, y=y, kind='scatter')` | 2 features + distributions |
| Bar (mean) | `sns.barplot(x='cat', y='num', data=df)` | Average comparison |
| Swarm | `sns.swarmplot(x='cat', y='num', data=df)` | Individual data points |
| Style | `sns.set_style('whitegrid')` | Set visual theme |
| Palette | `sns.set_palette('viridis')` | Set color scheme |

