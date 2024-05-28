from tensorflow.keras.preprocessing.text import Tokenizer

# Convert characters into one-hot encoded vectors
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(df['URL'])
X_one_hot = tokenizer.texts_to_matrix(df['URL'], mode='binary')