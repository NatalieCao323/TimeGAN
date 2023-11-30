import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, Input, Conv1D, LeakyReLU, Dropout, Flatten, Activation, Reshape, Bidirectional
from keras.optimizers import Adam
from scipy.stats import kurtosis
from pandas.plotting import autocorrelation_plot
import matplotlib.pyplot as plt

def load_data(filename, sequence_length):
    df = pd.read_csv(filename)
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(df.values)
    sequences = []
    for i in range(len(data_normalized) - sequence_length):
        sequences.append(data_normalized[i:i + sequence_length])
    return np.array(sequences), scaler

# Example usage
sequence_length = 60
data_normalized, scaler = load_data('stock_returns.csv', sequence_length)

def embedding_network(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(Dense(16, activation='relu'))  # Adjusted number of units
    return model

def recovery_network(latent_dim, input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(32, return_sequences=True), input_shape=(None, latent_dim)))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dense(input_shape[-1], activation='sigmoid'))
    return model

def generator(latent_dim, sequence_length):
    model = Sequential()
    model.add(Dense(sequence_length * 2, activation='relu', input_dim=latent_dim))  # Increased complexity
    model.add(Reshape((sequence_length, 2)))
    model.add(Conv1D(32, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(Conv1D(1, 3, padding='same'))
    model.add(Dropout(0.3))  # Added dropout
    return model

def discriminator(input_shape):
    model = Sequential()
    model.add(Conv1D(128, 3, padding='same', input_shape=input_shape))  # Increased filter count
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))  # Increased dropout
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

def train_timegan(sequences, latent_dim, epochs, batch_size):
    input_shape = sequences.shape[1:]
    emb_net = embedding_network(input_shape)
    rec_net = recovery_network(latent_dim, input_shape)
    gen_net = generator(latent_dim, sequence_length)
    dis_net = discriminator(input_shape)

    # Compile models with appropriate loss functions and optimizers
    # This needs to be filled in based on your model requirements

    for epoch in range(epochs):
        for i in range(0, len(sequences), batch_size):
            seq_batch = sequences[i:i + batch_size]
            # Implement training logic for each model component
            # This part requires detailed implementation based on TimeGAN methodology

            print(f"Epoch {epoch+1}, Batch {i+1}/{len(sequences)//batch_size}")

latent_dim = 100
gen_net = generator(latent_dim, sequence_length)
train_timegan(data_normalized, latent_dim, epochs=1000, batch_size=32)


def generate_synthetic_data(generator, latent_dim, num_samples):
    # Generate random points in the latent space
    random_latent_points = np.random.normal(0, 1, (num_samples, latent_dim))
    # Predict (generate) new data based on latent points
    synthetic_data = generator.predict(random_latent_points)
    return synthetic_data

num_samples = 1000  # Specify the number of synthetic samples you want to generate
synthetic_data = generate_synthetic_data(gen_net, latent_dim, num_samples)

synthetic_data_rescaled = scaler.inverse_transform(synthetic_data.reshape(-1, 1))
synthetic_data_rescaled = synthetic_data_rescaled.reshape(num_samples, -1)

def calculate_metrics(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    kurt = kurtosis(data, axis=None, fisher=True)  # Fisher's definition (normal ==> 0.0)
    return mean, std_dev, kurt

# Extract the first 10 samples from the real dataset
real_data_sample = data_normalized[:10, :, 0].squeeze()  # Adjust depending on your dataset's shape

# Extract the first 10 samples from the synthetic dataset
synthetic_data_sample = synthetic_data_rescaled[:10, :].squeeze()  # Adjust if necessary

# Calculate for real data
real_mean, real_std, real_kurt = calculate_metrics(real_data_sample)
# Calculate for synthetic data
synthetic_mean, synthetic_std, synthetic_kurt = calculate_metrics(synthetic_data_sample)


print("Real Data: Mean =", real_mean, ", Std Dev =", real_std, ", Kurtosis =", real_kurt)
print("Synthetic Data: Mean =", synthetic_mean, ", Std Dev =", synthetic_std, ", Kurtosis =", synthetic_kurt)

# Plotting the real data
plt.subplot(2, 1, 1)
plt.plot(data_normalized[:1000, :, 0].squeeze(), label='Real Data')  # Adjust as needed
plt.title('Real Stock Return Series')
plt.legend()

# Plotting the synthetic data
plt.subplot(2, 1, 2)
plt.plot(synthetic_data_rescaled[:1000], label='Synthetic Data')  # Ensure this is 2D
plt.title('Synthetic Stock Return Series')
plt.legend()

plt.tight_layout()
plt.show()

# Convert to pandas Series for easier plotting
real_series = pd.Series(real_data_sample.mean(axis=1))
synthetic_series = pd.Series(synthetic_data_sample.mean(axis=1))

# Plotting
plt.figure(figsize=(12, 6))


# Autocorrelation for real data
plt.subplot(2, 1, 1)
autocorrelation_plot(real_series)
plt.title('Autocorrelation of Real Data')

# Autocorrelation for synthetic data
plt.subplot(2, 1, 2)
autocorrelation_plot(synthetic_series)
plt.title('Autocorrelation of Synthetic Data')

plt.tight_layout()
plt.show()

np.savetxt('synthetic_stock_returns.csv', synthetic_data_rescaled, delimiter=',')
