# Function to convert URLs into sequences of ASCII values
def url_to_ascii_sequence(url):
    # Convert each character to its ASCII value
    return [ord(char) for char in url]

# Apply URL to ASCII sequence conversion
df['ASCII Sequence'] = df['URL'].apply(url_to_ascii_sequence)

# Convert sequences to numpy array
X_ascii = np.array(df['ASCII Sequence'].tolist())