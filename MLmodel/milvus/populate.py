import tensorflow_datasets as tfds
import tensorflow as tf
from pymilvus_orm import connections, Collection



"""
    SCRIPT TO POPULATE THE DATABASE WITH THE EMBEDDINGS
    OF THE REVIEWS AND THE LABELS
"""
# Connect to the default Milvus server
connections.connect()

# Get a reference to the Milvus collection
collection = Collection('Embeddings')

# Load the dataset and split it into train and test sets
dataset, info = tfds.load('imdb_reviews', split=('train', 'test'), as_supervised=True, with_info=True)

# Define the text vectorization layer
max_features = 10000
sequence_length = 200
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# Adapt the vectorization layer to the training set
text_ds = dataset[0].map(lambda x, y: x)
vectorize_layer.adapt(text_ds)

# Compute and store the vectorizations of the texts in the training set
count = 0
for example, label in dataset[0]:
    count += 1
    example = example.numpy()[0].decode('utf-8')
    example_vector = vectorize_layer(tf.expand_dims(example, -1))
    collection.insert([
        {'vector': example_vector.numpy()[0].tolist(), 'id': count}
    ])

# Print the number of rows in the collection
print("Number of rows in collection: ", collection.num_entities)
