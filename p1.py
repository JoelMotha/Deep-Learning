import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomUniform

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Model with tanh activation and more neurons
model = Sequential([
    Dense(4, input_dim=2, activation='tanh', kernel_initializer=RandomUniform(minval=-1, maxval=1)),
    Dense(1, activation='sigmoid', kernel_initializer=RandomUniform(minval=-1, maxval=1))
])

# Compile with a smaller learning rate
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train for more epochs
history = model.fit(X, y, epochs=5000, verbose=0)

# Predict and evaluate
predictions = model.predict(X)
predicted_classes = (predictions > 0.5).astype(int)

# Output results
print("Input:")
print(X)
print("Predicted Output:")
print(predicted_classes)
print("Expected Output:")
print(y)
