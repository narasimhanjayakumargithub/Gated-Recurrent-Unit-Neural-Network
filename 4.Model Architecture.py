from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, GRU, concatenate, Dense
from tensorflow.keras.models import Model

# Define input shape
input_shape = (max_length,)

# Define input layer
input_layer = Input(shape=input_shape)

# Embedding layer
embedding_dim = 50
embedding_layer = Embedding(input_dim=128, output_dim=embedding_dim)(input_layer)

# Convolutional layer
filters = 64
kernel_size = 3
conv_layer = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(embedding_layer)

# Max pooling layer
pooling_layer = MaxPooling1D(pool_size=2)(conv_layer)

# GRU layer
gru_units = 32
gru_layer = GRU(units=gru_units)(pooling_layer)

# Output layer
output_layer = Dense(1, activation='sigmoid')(gru_layer)

# Create model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
print(model.summary())