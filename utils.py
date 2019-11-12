import csv
import os
import pickle
import re
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds

path_to_dataset = 'cornell movie-dialogs corpus'

class Preprocess():

    def __init__(self, eval = False):
        self.path_to_movie_lines = os.path.join(path_to_dataset, 'movie_lines.txt')
        self.path_to_movie_conversations = os.path.join(path_to_dataset,
                                                   'movie_conversations.txt')
        self.wiki_questions, self.wiki_answers = self.load_conversations_wiki()
        self.ubuntu_questions, self.ubuntu_answers= self.load_conversations_ubuntu()
        self.cornell_questions, self.cornell_answers = self.load_conversations()
        self.fallout_questions, self.fallout_answers = self.load_conversations_fallout()
        # Build tokenizer using tfds for both questions and answers
        self.tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            self.wiki_questions +
            self.ubuntu_questions +
            self.fallout_questions +
            self.cornell_questions +
            self.wiki_answers +
            self.ubuntu_answers +
            self.fallout_answers +
            self.cornell_answers,
            target_vocab_size=2 ** 13)
        # Vocabulary size plus start and end token
        self.vocab_size = self.tokenizer.vocab_size + 2

        # Maximum sentence length
        self.MAX_LENGTH = 45

    def preprocess_sentence(self, sentence):
        sentence = sentence.lower().strip()
        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
        sentence = sentence.strip()
        # adding a start and an end token to the sentence
        return sentence


    # Tokenize, filter and pad sentences
    def tokenize_and_filter(self, inputs, outputs):
        # Define start and end token to indicate the start and end of a sentence
        START_TOKEN, END_TOKEN = [self.tokenizer.vocab_size], [self.tokenizer.vocab_size + 1]
        tokenized_inputs, tokenized_outputs = [], []

        for (sentence1, sentence2) in zip(inputs, outputs):
            # tokenize sentence
            sentence1 = START_TOKEN + self.tokenizer.encode(sentence1) + END_TOKEN
            sentence2 = START_TOKEN + self.tokenizer.encode(sentence2) + END_TOKEN
            # check tokenized sentence max length
            if len(sentence1) <= self.MAX_LENGTH and len(sentence2) <= self.MAX_LENGTH:
                tokenized_inputs.append(sentence1)
                tokenized_outputs.append(sentence2)

        # pad tokenized sentences
        tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_inputs, maxlen=self.MAX_LENGTH, padding='post')
        tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
            tokenized_outputs, maxlen=self.MAX_LENGTH, padding='post')

        return tokenized_inputs, tokenized_outputs

    def load_conversations(self):
      # dictionary of line id to text
      id2line = {}
      with open(self.path_to_movie_lines, errors='ignore') as file:
        lines = file.readlines()
      for line in lines:
        parts = line.replace('\n', '').split(' +++$+++ ')
        id2line[parts[0]] = parts[4]

      inputs, outputs = [], []
      with open(self.path_to_movie_conversations, 'r') as file:
        lines = file.readlines()
      for line in lines:
        parts = line.replace('\n', '').split(' +++$+++ ')
        # get conversation in a list of line ID
        conversation = [line[1:-1] for line in parts[3][1:-1].split(', ')]
        for i in range(len(conversation) - 1):
          inputs.append(self.preprocess_sentence(id2line[conversation[i]]))
          outputs.append(self.preprocess_sentence(id2line[conversation[i + 1]]))
      return inputs, outputs

    def load_conversations_fallout(self):
        with open('FalloutData/final.txt', 'r') as file:
            text = file.read()

        pairs = text.split('\n\n')
        inputs, outputs = [], []
        for pair in pairs:
            array = pair.split('\n')
            if len(array)==2:
                inputs.append(array[0])
                outputs.append(array[1])
        return inputs, outputs


    def load_conversations_ubuntu(self):
        return  pickle.load(open("ubuntuQA/prepprocessed_questions.p", "rb"))

    def load_conversations_wiki(self):
        df = pd.read_csv('wikiQA/WikiQA.tsv',  sep='\t')
        return  df['Question'].tolist(), df['Sentence'].tolist()



