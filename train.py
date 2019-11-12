import tensorflow as tf

import utils
from model import transformer

preprocessor = utils.Preprocess()

questions = preprocessor.wiki_questions + preprocessor.cornell_questions
answers = preprocessor.wiki_answers + preprocessor.cornell_answers
VOCAB_SIZE = preprocessor.vocab_size
questions, answers,  = preprocessor.tokenize_and_filter(questions, answers)

BATCH_SIZE = 32
BUFFER_SIZE = 20000

# remove START_TOKEN from targets
dataset = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions,
        'dec_inputs': answers[:, :-1]
    },
    {
        'outputs': answers[:, 1:]
    },
))

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

tf.keras.backend.clear_session()

# Hyper-parameters
NUM_LAYERS = 4
D_MODEL = 312
NUM_HEADS = 8
UNITS = 768
DROPOUT = 0.2

model = transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)


def loss_function(y_true, y_pred):
  y_true = tf.reshape(y_true, shape=(-1, preprocessor.MAX_LENGTH - 1))

  loss = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')(y_true, y_pred)

  mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
  loss = tf.multiply(loss, mask)

  return tf.reduce_mean(loss)


optimizer = tf.keras.optimizers.Adam(
    5e-3, beta_1=0.9, beta_2=0.98, epsilon=1e-7)

def accuracy(y_true, y_pred):
  # ensure labels have shape (batch_size, MAX_LENGTH - 1)
  y_true = tf.reshape(y_true, shape=(-1, preprocessor.MAX_LENGTH - 1))

  return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])

EPOCHS = 200

checkpoint_path = 'chkpts/chkpt'
model.load_weights(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
model.fit(dataset, epochs=EPOCHS, callbacks=[cp_callback])
