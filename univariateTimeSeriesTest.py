# Import required libraries
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

# Split a univariate sequence into samples

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):

        # find the end of this pattern
        end_ix = i + n_steps

        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break

        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)

    return array(X), array(y)

# Define input sequence
raw_seq = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650]

# Choose a number of time steps
n_steps = 3

# Split into samples
X, y = split_sequence(raw_seq, n_steps)

# Reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))

# Define 1-D CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation="relu", input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation="relu"))
model.add(Dense(1))
model.compile(optimizer='adam', loss="mse")

# Fit the model
model.fit(X, y, epochs=1000, verbose=0)

# Demonstrate prediction
x_input = array([150, 200, 250])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)
