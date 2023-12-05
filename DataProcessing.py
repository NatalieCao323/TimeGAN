import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('stock_returns.csv', low_memory=False)

# 1. Filter Data by Trading Volume
volume_percentiles = data.groupby('date')['VOL'].quantile(0.75)
data['top_25_volume'] = data.groupby('date')['VOL'].transform(lambda x: x >= volume_percentiles[x.name])
data.loc[~data['top_25_volume'], 'RETX'] = np.nan  # Set non-top 25% data to NaN

# Drop rows where RETX is NaN after filtering
data.dropna(subset=['RETX'], inplace=True)

# Filter outliers
data['RETX'] = pd.to_numeric(data['RETX'], errors='coerce')
data = data[data['RETX'].between(-1, 1)]

# 2. Create Windowed Data Samples
window_size = 60
samples = []
targets = []

for start in range(len(data) - window_size):
    end = start + window_size
    sample = data.iloc[start:end]  # Get the 60-day window
    target = data.iloc[end]  # Get the 61st day

    # Check for NaN values in the window
    if not sample['RETX'].isnull().values.any():
        samples.append(sample['RETX'].values)
        if not np.isnan(target['RETX']):
            targets.append(target['RETX'])  # Append the target value
        else:
            targets.append(np.nan)  # Append NaN if the target value is missing


# Convert to numpy arrays
samples = np.array(samples)
targets = np.array(targets)

# 3. Normalize data
scaler = StandardScaler()
samples_scaled = scaler.fit_transform(samples)

# Flatten the 2D array back to 1D and use it as inputs for VAE
inputs = samples_scaled.reshape(samples_scaled.shape[0], -1)

# Export the scaler, mean, and standard deviation
mean_value = scaler.mean_[0]
std_dev_value = np.sqrt(scaler.var_[0])

# Now `inputs` is a numpy array that you can pass into the VAE