import matplotlib.pyplot as plt
import pandas as pd
import math
from matplotlib.ticker import MultipleLocator
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("metrics_file", type=str, help="Path to the metrics CSV file")
parser.add_argument("--start_iter", type=int, default=500, help="Start iteration for slicing")
parser.add_argument("--end_iter", type=int, default=None, help="End iteration for slicing")
parser.add_argument("--start_run", type=int, default=None, help="Start run for slicing")
parser.add_argument("--end_run", type=int, default=10, help="End run for slicing")    

args = parser.parse_args()

df = pd.read_csv(args.metrics_file)

assert "run" in df.columns, "Column 'run' not found in the metrics file"
assert "iteration" in df.columns, "Column 'iteration' not found in the metrics file"

# list all cols that ends with _ns
metrics = [col for col in df.columns if col.endswith("_ns")]
# print(f"Found {len(metrics)} metrics: {metrics}")

# create a sub dataframe with run, curr_tree_cost
# where curr_tree_cost is the max aggregated by run

if args.start_iter is not None:
    df = df[df["iteration"] >= args.start_iter]
if args.end_iter is not None:
    df = df[df["iteration"] <= args.end_iter]

if args.start_run is not None:
    df = df[df["run"] >= args.start_run]
if args.end_run is not None:
    df = df[df["run"] <= args.end_run]
    
df_grouped = df.groupby("iteration").mean().reset_index()
df_grouped = df_grouped.drop(columns=["run"])

df_grouped["total_ns"] = df_grouped[metrics].sum(axis=1)

metrics.append("total_ns")

# print(df_grouped.head())
df_grouped["node_size_max"] = (
    df_grouped["total_ns"].rolling(window=100, min_periods=1).max()
)

# compute cumulative for each metric
for m in metrics:
    df_grouped[m + "_cumulative"] = df_grouped[m].cumsum()

# cumulative metrics list
cumulative_metrics = [m + "_cumulative" for m in metrics]
cumulative_metrics.append("node_size")
cumulative_metrics.append("node_size_max")

# subplot grid: 2 columns
ncols = 2
nrows = math.ceil((len(cumulative_metrics) + 1) / ncols)

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(12, 3 * nrows),
    sharex=False,
)

axes = axes.flatten()

for ax, metric in zip(axes[:len(cumulative_metrics)], cumulative_metrics):
    ax.plot(df_grouped["iteration"], df_grouped[metric] / 1000 if metric.endswith("_ns_cumulative") else df_grouped[metric])
    # ax.axvline(x=features * 10, color="red", linestyle="--", linewidth=1)  # vertical line

    max_value = df_grouped[metric].max() / 1000 if metric.endswith("_ns_cumulative") else df_grouped[metric].max()
    ax.axhline(y=max_value, color="red", linestyle="--", linewidth=1)
    ax.text(0.95, 0.90, f"Max: {max_value:.3f}", transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', color="red",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="red", facecolor="white", alpha=0.8))
    
    # plot avg line
    avg_value = df_grouped[metric].mean() / 1000 if metric.endswith("_ns_cumulative") else df_grouped[metric].mean()
    ax.axhline(y=avg_value, color="green", linestyle="--", linewidth=1)
    ax.text(0.95, 0.80, f"Avg: {avg_value:.3f}", transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', color="green",
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="green", facecolor="white", alpha=0.8))       

    ax.set_title(metric.replace("_", " ").title())
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cumulative Duration (µs)")

    # ax.xaxis.set_major_locator(MultipleLocator(1000))
    # ax.tick_params(axis='x', rotation=45)

    ax.ticklabel_format(style='plain', axis='y')

    ax.grid(True)

# # Plot cost vs run 
ax = axes[len(cumulative_metrics)]
# plot on x axis run,iter and on y axis curr_tree_cost
max_iter = df["iteration"].max()
min_iter = df["iteration"].min()
iters_per_run = max_iter - min_iter + 1

max_run = df["run"].max()
min_run = df["run"].min()

# ax.plot(df_tree["run"] + df_tree["run"].max() * (df_tree.index / len(df_tree)), df_tree["curr_tree_cost"])
# ax.plot(df_tree["run"], df_tree["curr_tree_cost"], marker='o')
# add a vertical line every run

runs = np.arange(min_run, max_run + 1)
for run in runs:
    ax.axvline(x=run * iters_per_run, color="gray", linestyle="--", linewidth=0.5)

# order df by run, iteration
df = df.sort_values(by=["run", "iteration"])
ax.plot(df["run"] * iters_per_run + (df["iteration"] - min_iter), df["curr_tree_cost"])

ax.set_title("Current Tree Cost vs Run, iter")
ax.set_xlabel("Run")
ax.set_ylabel("Current Tree Cost")
# make x ticks as run number
ax.set_xticks(runs * iters_per_run)
ax.set_xticklabels(runs)
ax.grid(True)




# Add easy slicing with two variables: start_iter and end_iter
# start_iter = 200  # e.g., 100
# end_iter = None    # e.g., 1000

# window_size = 100  # You can change this value

# for ax, metric in zip(axes, metrics):
#     # Apply mask per plot
#     mask = pd.Series([True] * len(df_grouped))
#     if start_iter is not None:
#         mask &= df_grouped["iteration"] >= start_iter
#     if end_iter is not None:
#         mask &= df_grouped["iteration"] <= end_iter

#     values = df_grouped.loc[mask, metric] / 1000
#     iterations = df_grouped.loc[mask, "iteration"]
#     windowed_avg = values.rolling(window=window_size, min_periods=1).mean()

#     ax.plot(iterations, windowed_avg)
#     ax.set_title(f"{metric.replace('_', ' ').title()} (Sliding Window Avg)")
#     ax.set_xlabel("Iteration")
#     ax.set_ylabel("Cumulative Duration (µs)")
#     ax.grid(True)

# remove empty subplots if any
for ax in axes[len(cumulative_metrics) + 1:]:
    ax.axis("off")

fig.suptitle(args.metrics_file, fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


