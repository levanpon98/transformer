import tensorflow as tf
from sklearn.model_selection import train_test_split
from model.Transformer import Transformer
from utils.preprocess import *
from utils.masking import *

path_input = 'D:/Kaggle/data/viet-eng/train.en'
path_target = 'D:/Kaggle/data/viet-eng/train.vi'
num_examples = 133317

BATCH_SIZE = 64
BUFFER_SIZE = 20000

# Try experimenting with the size of that dataset
input_tensor, target_tensor, inp_lang, target_lang = load_data(path_input, path_target, num_examples)

max_length_target, max_length_inp = max_length(target_tensor), max_length(input_tensor)

# Creating training and validation sets using an 80-20 split
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                target_tensor,
                                                                                                test_size=0.2)

steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
embedding_dim = 256
units = 1024
input_vocab_size = len(inp_lang.word_index) + 2
target_vocab_size = len(target_lang.word_index) + 2
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size,
                          pe_input=input_vocab_size,
                          pe_target=target_vocab_size,
                          rate=dropout_rate)


def evaluate(inp_sentence):
    start_token = [inp_lang.vocab_size]
    end_token = [inp_lang.vocab_size + 1]

    sentence = preprocess_sentence(inp_sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')

    inputs = tf.convert_to_tensor(inputs)

    #
    # inp sentence is portuguese, hence adding the start and end token
    inp_sentence = start_token + inputs + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    # as the target is english, the first word to the transformer should be the
    # english start token.
    decoder_input = [target_lang.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(max_length_target):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if predicted_id == target_lang.vocab_size + 1:
            return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


def translate(sentence, plot=''):
    result, attention_weights = evaluate(sentence)

    predicted_sentence = target_lang.decode([i for i in result
                                              if i < target_lang.vocab_size])

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(predicted_sentence))
