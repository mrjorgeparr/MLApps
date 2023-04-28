import tensorflow_datasets as tfds

# Load the dataset and split it into train and test sets
dataset, info = tfds.load('imdb_reviews', split=('train', 'test'), as_supervised=True,  with_info=True)
print("*****INFO*****\n")
print(info)
print("*****DATASET****\n")
print(dataset)


count = 0
for example, label in dataset[0].take(5):
    count += 1
    print("Label of example ", count, ": ", label.numpy())