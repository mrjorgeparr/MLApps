import tensorflow_datasets as tfds

# Load the IMDB reviews dataset in supervised mode
dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

# Split the dataset into training and testing
train_dataset, test_dataset = dataset['train'], dataset['test']

# Convert the TensorFlow tensors to numpy arrays and access the inputs
train_examples = [example[0] for example in tfds.as_numpy(train_dataset)]
test_examples = [example[0] for example in tfds.as_numpy(test_dataset)]

# Print the first example
print('Vectorized representation of the first example:\n', train_examples[1])
