# import .csv from <path>
# format: start_idx,end_idx,cost,n_splits

import argparse
import os

import pandas as pd
import numpy as np
from matplotlib import cm
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str, help="Path to directory containing leaf dump CSV files")
args = parser.parse_args()

path = args.path
files = [f for f in os.listdir(path) if f.endswith(".csv")]
files.sort()

# start_idx,end_idx,cost,n_splits
dfs = []
for f in files:
    df = pd.read_csv(os.path.join(path, f))
    # Add file identifier
    df['file'] = f
    dfs.append(df)

# Concatenate all dataframes
all_data = pd.concat(dfs, ignore_index=True)
all_data['size'] = all_data['end_idx'] - all_data['start_idx']

# Normalize costs by file (min-max normalization within each file)
cost_norms = []
for f in files:
    file_data = all_data[all_data['file'] == f]
    cost_min = file_data['cost'].min()
    cost_max = file_data['cost'].max()
    normalized_cost = (file_data['cost'] - cost_min) / (cost_max - cost_min)
    cost_norms.append(normalized_cost)

all_data['cost_normalized'] = pd.concat(cost_norms, ignore_index=True)

# Calculate correlations
size_n_splits_corr = all_data['size'].corr(all_data['n_splits'])
cost_n_splits_corr = all_data['cost_normalized'].corr(all_data['n_splits'])
size_cost_corr = all_data['size'].corr(all_data['cost_normalized'])

# n_splits distribution fitting
n_splits_data = all_data['n_splits'].values

# Method 1: Log-normal distribution (common for count data)
# Take log of data (add small epsilon to avoid log(0))
n_splits_log = np.log(n_splits_data + 1e-6)

# Fit normal distribution to log-transformed data
mu, std = stats.norm.fit(n_splits_log)
lognormal_params = (mu, std)

# Method 2: Poisson distribution (alternative for count data)
lambda_poisson = np.mean(n_splits_data)
poisson_params = lambda_poisson

# Method 3: Gamma distribution (also good for skewed positive data)
gamma_shape, gamma_scale, _ = stats.gamma.fit(n_splits_data, floc=0)
gamma_params = (gamma_shape, gamma_scale)

# Create color map for files
n_files = len(files)
colors = plt.cm.Set1(np.linspace(0, 1, n_files))
file_colors = dict(zip(files, colors))

print(f"Found {len(files)} files in {path}")
print(f"Total samples: {len(all_data)}")
print(f"n_splits statistics:")
print(f"  Mean: {all_data['n_splits'].mean():.2f}")
print(f"  Median: {all_data['n_splits'].median():.2f}")
print(f"  Std: {all_data['n_splits'].std():.2f}")
print(f"  Skewness: {all_data['n_splits'].skew():.3f}")
print(f"Distribution approximations:")
print(f"  Lognormal (μ, σ): ({lognormal_params[0]:.3f}, {lognormal_params[1]:.3f})")
print(f"  Poisson (λ): {poisson_params:.3f}")
print(f"  Gamma (α, θ): ({gamma_params[0]:.3f}, {gamma_params[1]:.3f})")
print(f"Correlations:")
print(f"  Size vs n_splits: {size_n_splits_corr:.3f}")
print(f"  Cost vs n_splits: {cost_n_splits_corr:.3f}")
print(f"  Size vs Cost: {size_cost_corr:.3f}")

plt.figure(figsize=(18, 14))

# 1. Normalized Cost vs Size
plt.subplot(2, 3, 1)
df_sorted_size = all_data.sort_values(by="size")
for f in files:
    file_data = df_sorted_size[df_sorted_size['file'] == f]
    if not file_data.empty:
        color = file_colors[f]
        plt.scatter(file_data["size"], file_data["cost_normalized"], 
                    label=f.replace('.csv', ''), alpha=0.6, s=30, color=color)

# Add correlation line
z = np.polyfit(all_data["size"], all_data["cost_normalized"], 1)
p = np.poly1d(z)
plt.plot(all_data["size"], p(all_data["size"]), "r--", alpha=0.8, linewidth=2)

plt.xlabel("Size (end_idx - start_idx)")
plt.ylabel("Normalized Cost")
plt.title("Normalized Cost vs Size")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()

# 2. Size vs n_splits (correlation plot)
plt.subplot(2, 3, 2)
df_sorted_size_nsplits = all_data.sort_values(by="size")
for f in files:
    file_data = df_sorted_size_nsplits[df_sorted_size_nsplits['file'] == f]
    if not file_data.empty:
        color = file_colors[f]
        plt.scatter(file_data["size"], file_data["n_splits"], 
                    label=f.replace('.csv', ''), alpha=0.6, s=30, color=color)

# Add correlation line
z_ns = np.polyfit(all_data["size"], all_data["n_splits"], 1)
p_ns = np.poly1d(z_ns)
plt.plot(all_data["size"], p_ns(all_data["size"]), "r--", alpha=0.8, linewidth=2)

plt.xlabel("Size (end_idx - start_idx)")
plt.ylabel("n_splits")
plt.title(f"Size vs n_splits (r={size_n_splits_corr:.3f})")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()

# 3. Size Distribution
plt.subplot(2, 3, 3)
size_counts = all_data['size'].value_counts().sort_index()
bars = plt.bar(size_counts.index, size_counts.values, alpha=0.7, color='skyblue')
plt.xlabel("Size (end_idx - start_idx)")
plt.ylabel("Count")
plt.title("Size Distribution (All Files)")
plt.grid(axis='y', alpha=0.3)

# Set denser x-axis ticks to show all unique sizes
plt.xticks(ticks=size_counts.index)
plt.gca().set_xticklabels(size_counts.index, rotation=45, ha='right')

# 4. Normalized Cost vs n_splits (correlation plot)
plt.subplot(2, 3, 4)
df_sorted_n_splits = all_data.sort_values(by="n_splits")
for f in files:
    file_data = df_sorted_n_splits[df_sorted_n_splits['file'] == f]
    if not file_data.empty:
        color = file_colors[f]
        plt.scatter(file_data["n_splits"], file_data["cost_normalized"], 
                    label=f.replace('.csv', ''), alpha=0.6, s=30, color=color)

# Add correlation line
z_cs = np.polyfit(all_data["n_splits"], all_data["cost_normalized"], 1)
p_cs = np.poly1d(z_cs)
plt.plot(all_data["n_splits"], p_cs(all_data["n_splits"]), "r--", alpha=0.8, linewidth=2)

plt.xlabel("n_splits")
plt.ylabel("Normalized Cost")
plt.title(f"Normalized Cost vs n_splits (r={cost_n_splits_corr:.3f})")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid()

# 5. Enhanced n_splits Distribution with multiple approximations
plt.subplot(2, 3, 5)
nsplits_counts = all_data['n_splits'].value_counts().sort_index()
x = nsplits_counts.index.astype(float)
y = nsplits_counts.values
bars_nsplits = plt.bar(x, y, alpha=0.7, color='lightcoral', label='Observed')
plt.xlabel("n_splits")
plt.ylabel("Count")
plt.title("n_splits Distribution (All Files)")
plt.grid(axis='y', alpha=0.3)

# Calculate x-range for fitted distributions
x_range = np.linspace(x.min(), x.max(), 100)

# Plot lognormal approximation (best for skewed counts)
lognormal_fit = stats.lognorm.pdf(x_range, s=lognormal_params[1], scale=np.exp(lognormal_params[0]))
plt.plot(x_range, len(n_splits_data) * lognormal_fit, 'b-', linewidth=2, 
         label=f'Lognormal fit (μ={lognormal_params[0]:.2f}, σ={lognormal_params[1]:.2f})')

# Plot Poisson approximation
poisson_fit = stats.poisson.pmf(x_range.astype(int), poisson_params)
plt.plot(x_range, len(n_splits_data) * poisson_fit, 'g--', linewidth=2, 
         label=f'Poisson fit (λ={poisson_params:.2f})')

# Plot Gamma approximation
gamma_fit = stats.gamma.pdf(x_range, gamma_params[0], scale=gamma_params[1])
plt.plot(x_range, len(n_splits_data) * gamma_fit, 'm:', linewidth=2, 
         label=f'Gamma fit (α={gamma_params[0]:.2f}, θ={gamma_params[1]:.2f})')

# Add mean line
mean_nsplits = all_data['n_splits'].mean()
plt.axvline(x=mean_nsplits, color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {mean_nsplits:.2f}')

# Add reference line at x=16 (perfect balanced tree depth)
plt.axvline(x=16, color='green', linestyle='-', linewidth=2, 
            label='Perfect balance (depth=16)')

plt.legend()
plt.xticks(rotation=45, ha='right')

# 6. Correlation Matrix Heatmap
plt.subplot(2, 3, 6)
correlation_data = all_data[['size', 'n_splits', 'cost_normalized']].corr()
sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title("Correlation Matrix")

plt.tight_layout()
plt.show()

# Print goodness of fit statistics
print("\nGoodness of fit (by Kolmogorov-Smirnov test):")
print(f"  Lognormal: {stats.kstest(all_data['n_splits'], 'lognorm', args=lognormal_params)[1]:.4f}")
print(f"  Poisson: {stats.kstest(all_data['n_splits'], 'poisson', args=[poisson_params])[1]:.4f}")
print(f"  Gamma: {stats.kstest(all_data['n_splits'], 'gamma', args=gamma_params)[1]:.4f}")
print("Higher p-values indicate better fit (p > 0.05 typically considered good)")

# Additional analysis: Proportion beyond depth 16
proportion_beyond_16 = (all_data['n_splits'] >= 16).mean()
proportion_at_16 = (all_data['n_splits'] == 16).mean()
print(f"\nTree depth analysis relative to balanced depth 16:")
print(f"  Proportion of leaves with ≥16 splits (depth ≥4): {proportion_beyond_16:.3f}")
print(f"  Proportion of leaves with exactly 16 splits: {proportion_at_16:.3f}")
print(f"  Mean n_splits vs target 16: {all_data['n_splits'].mean():.2f} vs 16.0")