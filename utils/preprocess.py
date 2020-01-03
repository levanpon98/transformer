import tensorflow as tf

import io
import re
import unicodedata
import tensorflow_datasets as tfds


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

    return tensor, lang_tokenizer


def preprocess_sentence(sentence):
    sentence = sentence.lower()
    # sentence = re.sub(r' .', '.', sentence)
    # sentence = re.sub(r' , ', ', ', sentence)
    # sentence = re.sub(r'&apos;', '\'', sentence)
    # sentence = re.sub(r'&quot;', '"', sentence)
    sentence = sentence.strip().rstrip()
    # sentence = '<start>' + sentence + '<end>'

    return sentence


def max_length(tensor):
    return max(len(t) for t in tensor)


def create_dataset(path_input, path_target, num_examples=None):
    lines_input = io.open(path_input, encoding='UTF-8').read().strip().split('\n')
    lines_target = io.open(path_target, encoding='UTF-8').read().strip().split('\n')

    inp = [preprocess_sentence(sentence) for sentence in lines_input[:num_examples]]
    target = [preprocess_sentence(sentence) for sentence in lines_target[:num_examples]]

    return inp, target


def load_data(path_input, path_target, num_examples=None):
    target_lang, inp_lang = create_dataset(path_input, path_target, num_examples)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, target_lang_tokenizer = tokenize(target_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, target_lang_tokenizer
