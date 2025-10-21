import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

plot_dir = "./plots/"

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

trial_name = "t1"  # Change this to the desired trial name (e.g., t1, t2, bt1, bt2)

SYNCHED_CSV_PATH = f"/standard/UVA-DSA/MIDAS/Organized/final_data/{trial_name}/synched_data/synced_all_{trial_name}.csv"  # <-- set your combined CSV path
# SYNCHED_CSV_PATH = f"/standard/UVA-DSA/MIDAS/Organized/final_data/{trial_name}/synched_data/raven_console_combined_{trial_name}.csv"
print(f"Loading data from: {SYNCHED_CSV_PATH}")

combined = pd.read_csv(SYNCHED_CSV_PATH)

x_axis = "obs_frame_idx"  # x-axis for plots
raven_var = "raven_field.pos0"
console_var = "console_pos0"

period_start = 4200  # in frames
period_end = 5200    # in frames



# columns of interest
columns_of_interest = [x_axis, raven_var, console_var]

# obs_frame_idx is x-axis
# Plot raven_field.pos0 and console_pos0 over obs_frame_idx

# Extract relevant columns
data = combined[columns_of_interest].copy()

# bring raven_field.pos0 and console_pos0 to similar scale for better visualization
data[raven_var] = (data[raven_var] - data[raven_var].min()) / (data[raven_var].max() - data[raven_var].min())
data[console_var] = (data[console_var] - data[console_var].min()) / (data[console_var].max() - data[console_var].min())

# Focus on the specified period
data = data[(data[x_axis] >= period_start) & (data[x_axis] <= period_end)]

# Plotting
plt.figure(figsize=(15, 7))
for modality in columns_of_interest[1:]:
    plt.plot(data[x_axis], data[modality], label=modality)
plt.xlabel("Frame Number")
plt.ylabel("Position Value")
plt.title(f"Raven {raven_var} and Console {console_var} over Frame Number")
plt.legend()
plt.grid()

plt.savefig(f"./plots/{trial_name}_{raven_var}_{console_var}_over_time.png")
