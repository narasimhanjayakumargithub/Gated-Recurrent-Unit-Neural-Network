import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Sample data (replace with your dataset)
data = {
    'URL': ['http://example.com', 'http://malicious.com', 'https://legit.com', 'http://phishing.net'],
    'Label': [0, 1, 0, 1]  # 0: benign, 1: malicious
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Function to convert URLs into sequences of characters
def url_to_sequence(url):
    # You may use different tokenization methods here
    return [char for char in url]

# Apply URL to sequence conversion
df['Sequence'] = df['URL'].apply(url_to_sequence)

# Determine max sequence length
max_length = max(df['Sequence'].apply(len))

# Padding sequences to make them equal length
df['Padded Sequence'] = df['Sequence'].apply(lambda x: x + [0] * (max_length - len(x)))

# Convert sequences to numpy array
X = np.array(df['Padded Sequence'].tolist())
y = np.array(df['Label'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print sample data
print("Sample Data:")
print(df[['URL', 'Sequence', 'Padded Sequence', 'Label']])