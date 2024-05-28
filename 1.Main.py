import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, GRU, concatenate, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

# Assume you have a dataset with URLs and their labels (0 for benign, 1 for malicious)
# X contains URLs and y contains labels
# You need to preprocess your data to convert URLs to sequences of integers

# Example preprocessing function (you may need to customize this based on your dataset)
def preprocess_data(urls, max_length):
    # Convert URLs to sequences of integers (using some tokenization method)
    # Example tokenization: convert each character to its ASCII value
    sequences = [[ord(char) for char in url] for url in urls]
    
    # Pad sequences to a fixed length
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    
    return padded_sequences

# Example dataset
urls = ['http://example.com', 'http://malicious.com', 'https://legit.com', 'http://phishing.net']
labels = [0, 1, 0, 1]  # 0: benign, 1: malicious

# Preprocess data
max_length = 50  # Max length of URL
X = preprocess_data(urls, max_length)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Define model architecture
embedding_dim = 50
input_layer = Input(shape=(max_length,))
embedding_layer = Embedding(input_dim=128, output_dim=embedding_dim)(input_layer)
conv_layer = Conv1D(filters=64, kernel_size=3, activation='relu')(embedding_layer)
pooling_layer = MaxPooling1D(pool_size=2)(conv_layer)
gru_layer = GRU(units=32)(pooling_layer)
output_layer = Dense(1, activation='sigmoid')(gru_layer)

# Create model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
print(model.summary())

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)