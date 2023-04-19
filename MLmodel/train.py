import keras_nlp
from tensorflow import keras
import tensorflow_datasets as tfds



imdb_train, imdb_test = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    as_supervised=True,
    batch_size=16)

classifier = keras_nlp.models.BertClassifier.from_preset(
    "bert_base_en_uncased"
)

classifier.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.Adam(5e-5),
    metrics = keras.metrics.SparseCategoricalAccuracy(),
    jit_compile=True,
)

history = classifier.fit(
    imdb_train,
    validation_data=imdb_test,
    epochs=1,
)

"""
Missing computing more evaluation metrics, from history.
"""
# classifier.predict(f"that movie was dog shit")

