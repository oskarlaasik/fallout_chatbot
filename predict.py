import tensorflow as tf
import utils
from model import transformer
preprocessor = utils.Preprocess()


questions = preprocessor.wiki_questions + preprocessor.cornell_questions + preprocessor.ubuntu_questions
answers = preprocessor.wiki_answers + preprocessor.cornell_answers + preprocessor.ubuntu_answers
VOCAB_SIZE = preprocessor.vocab_size

questions, answers = preprocessor.tokenize_and_filter(questions, answers)

# Hyper-parameters
NUM_LAYERS = 4
D_MODEL = 312
NUM_HEADS = 8
UNITS = 768
DROPOUT = 0.1

model = transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)

model.load_weights('chkpts/chkpt')


def evaluate(sentence):
    sentence = preprocessor.preprocess_sentence(sentence)
    # Build tokenizer using tfds for both questions and answers
    START_TOKEN, END_TOKEN = [preprocessor.tokenizer.vocab_size], [preprocessor.tokenizer.vocab_size + 1]

    sentence = tf.expand_dims(
        START_TOKEN + preprocessor.tokenizer.encode(sentence) + END_TOKEN, axis=0)

    output = tf.expand_dims(START_TOKEN, 0)

    for i in range(preprocessor.MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        # concatenated the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0)


def predict(sentence):
    prediction = evaluate(sentence)

    predicted_sentence = preprocessor.tokenizer.decode(
        [i for i in prediction if i < preprocessor.tokenizer.vocab_size])

    print('Input: {}'.format(sentence))
    print('Output: {}'.format(predicted_sentence))

    return predicted_sentence

while True:
    predict(input())