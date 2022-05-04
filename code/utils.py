# This file is an excerpt from the following file with minor adjustments:
# https://github.com/francois-meyer/lexical-ambiguity-dms/blob/master/code/src/utils.py
# which belongs to the following paper:
# Meyer, F., & Lewis, M. (2020). Modelling lexical ambiguity with density matrices. arXiv preprint arXiv:2010.05670.

from torchtext.legacy.data import Dataset
import numpy as np
import torch
from collections import Counter
import math
import random


class SkipGramVocab:
    def __init__(self, corpus_path, min_count, subsampling, neg_table_size):
        self.corpus_path = corpus_path
        self.min_count = min_count
        self.subsampling = subsampling
        self.neg_table_size = neg_table_size

        self.num_sentences = 0
        self.train_count = 0
        self.counter = Counter()
        self.stoi = {}
        self.itos = []
        self.subsampling_probs = {}
        self.neg_dist = None
        self.neg_table = []

    def build_vocab(self):
        self.count_words()
        self.apply_min_count()
        self.setup_negsampling()
        self.setup_subsampling()

    def count_words(self):
        with open(self.corpus_path, encoding="latin1") as file:
            for line in file:
                tokens = line.split()
                if len(tokens) > 0:
                    self.num_sentences += 1
                for token in tokens:
                    self.counter.update([token])
        print("Raw vocab size is %d." % len(self.counter))

    def apply_min_count(self):
        discard_count = 0
        raw_train_count = 0
        for word in list(self.counter):
            raw_train_count += self.counter[word]
            if self.counter[word] < self.min_count:
                del self.counter[word]
                discard_count += 1
            else:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)
                self.train_count += self.counter[word]

        print(
            "Discarded %d words with count less than %d."
            % (discard_count, self.min_count)
        )
        print("Final vocab size is %d." % len(self.itos))
        print(
            "Training words down from %d to %d." % (raw_train_count, self.train_count)
        )

    def setup_negsampling(self):
        # Compute distribution for negative sampling
        self.neg_dist = np.array(list(self.counter.values()))
        self.neg_dist = np.power(self.neg_dist, 0.75)
        self.neg_dist = self.neg_dist / self.neg_dist.sum()

        self.neg_dist = np.round(self.neg_dist * self.neg_table_size)
        for word_index, count in enumerate(self.neg_dist):
            self.neg_table += [word_index] * int(count)
        self.neg_dist = None  # free up memory
        self.neg_table = np.array(self.neg_table)
        np.random.shuffle(self.neg_table)

    def setup_subsampling(self):
        subsampling_count = 0
        threshold_count = self.train_count * self.subsampling
        print("Subsampling count threshold is %d." % threshold_count)

        for word in self.itos:
            sample_prob = (math.sqrt(self.counter[word] / threshold_count) + 1) * (
                threshold_count / self.counter[word]
            )
            self.subsampling_probs[word] = min(sample_prob, 1.0)
            if sample_prob < 1.0:
                subsampling_count += 1
        print(
            "Subsampling %d words with frequency greater than %f."
            % (subsampling_count, self.subsampling)
        )

    def contains(self, word):
        return word in self.stoi

    def get_index(self, word):
        return self.stoi[word]

    def size(self):
        return len(self.itos)


class Word2DMVocab:
    def __init__(self, corpus_path, min_count, subsampling, neg_table_size):
        self.corpus_path = corpus_path
        self.min_count = min_count
        self.subsampling = subsampling
        self.neg_table_size = neg_table_size
        self.num_sentences = 0
        # number of words in text usable for training after discarding those that occur less often than min_count
        self.train_count = 0
        # number of occurences per word
        self.counter = Counter()
        # indices of words in one-hot encoding
        self.stoi = {}
        # list of words in vocab
        self.itos = []
        self.subsampling_probs = {}
        self.neg_dist = None
        self.neg_table = []

    def build_vocab(self):
        self.count_words()
        self.apply_min_count()
        self.setup_negsampling()
        self.setup_subsampling()

    def count_words(self):
        with open(self.corpus_path, encoding="latin1") as file:
            for line in file:
                tokens = line.split()
                if len(tokens) > 0:
                    self.num_sentences += 1
                for token in tokens:
                    self.counter.update([token])
        print("Raw vocab size is %d." % len(self.counter))

    def apply_min_count(self):
        discard_count = 0
        raw_train_count = 0
        for word in list(self.counter):
            raw_train_count += self.counter[word]
            if self.counter[word] < self.min_count:
                del self.counter[word]
                discard_count += 1
            else:
                self.stoi[word] = len(self.itos)
                self.itos.append(word)
                self.train_count += self.counter[word]

        print(
            "Discarded %d words with count less than %d."
            % (discard_count, self.min_count)
        )
        print("Final vocab size is %d." % len(self.itos))
        print(
            "Training words down from %d to %d." % (raw_train_count, self.train_count)
        )

    def setup_negsampling(self):
        # Compute distribution for negative sampling
        # builds a shuffled table, which has maximal neg_table_size entries, where the number of occurences of the index of a word is proportional to its number of counts
        # array of counts of words
        # http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
        self.neg_dist = np.array(list(self.counter.values()))
        # counts of words to the power of 0.75 (authors found that this works best)
        self.neg_dist = np.power(self.neg_dist, 0.75)
        # normalize --> proability for a word occuring in the text
        self.neg_dist = self.neg_dist / self.neg_dist.sum()
        # multiply by neg_table_size and round
        self.neg_dist = np.round(self.neg_dist * self.neg_table_size)
        #  append the word_index count times to the list
        for word_index, count in enumerate(self.neg_dist):
            self.neg_table += [word_index] * int(count)
        self.neg_dist = None  # free up memory
        self.neg_table = np.array(self.neg_table)
        # shuffle the array
        np.random.shuffle(self.neg_table)

    def setup_subsampling(self):
        # some words that occur very often are discarded, e.g. the. words are kept with self.sub_sampling_probs[word]
        subsampling_count = 0
        threshold_count = self.train_count * self.subsampling
        print("Subsampling count threshold is %d." % threshold_count)

        for word in self.itos:
            # probability of keeping the word
            # http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
            sample_prob = (math.sqrt(self.counter[word] / threshold_count) + 1) * (
                threshold_count / self.counter[word]
            )
            self.subsampling_probs[word] = min(sample_prob, 1.0)
            if sample_prob < 1.0:
                subsampling_count += 1
        print(
            "Subsampling %d words with frequency greater than %f."
            % (subsampling_count, self.subsampling)
        )

    def contains(self, word):
        return word in self.stoi

    def get_index(self, word):
        return self.stoi[word]

    def size(self):
        return len(self.itos)


class SkipGramDataset(Dataset):
    def __init__(self, corpus_path, vocab, window_size, neg_samples):
        self.window_size = window_size
        self.corpus_file = open(corpus_path, encoding="latin1")
        self.vocab = vocab
        self.vocab_size = len(vocab.itos)
        self.corpus_size = vocab.num_sentences
        self.neg_samples = neg_samples
        self.neg_index = 0

    def __len__(self):
        return self.corpus_size

    def __getitem__(self, idx):
        # build lists with word idx per sentence in corpus, apply subsampling, return sgns_predictions of these idx lists
        while True:
            line = self.corpus_file.readline()
            if not line:
                self.corpus_file.seek(0, 0)
                line = self.corpus_file.readline()
            if len(line) > 1:
                words = line.split()
                if len(words) > 1:
                    word_ids = [
                        self.vocab.stoi[word]
                        for word in words
                        if word in self.vocab.stoi
                        and random.random() < self.vocab.subsampling_probs[word]
                    ]
                    return self.generate_sgns_predictions(word_ids)

    # sgns: skip-gram with negative sampling
    def generate_sgns_predictions(self, word_ids):
        if len(word_ids) > 100 or len(word_ids) <= 1:
            return [], [], torch.LongTensor()

        dynamic_window_size = int(random.random() * self.window_size) + 1
        target_ids = []
        context_ids = []
        neg_ids = []
        for i, target_id in enumerate(word_ids):
            window_ids = word_ids[
                max(0, i - dynamic_window_size) : min(
                    len(word_ids), i + dynamic_window_size + 1
                )
            ]
            window_ids = [word for word in window_ids if word != target_id]
            target_ids.extend([target_id] * len(window_ids))
            context_ids.extend(window_ids)

        if len(target_ids) == 0:
            return [], [], torch.LongTensor()
        ###################################################################################################
        # adjust neg sampling such that neg samples can not be the same as target and context word
        for i, t in enumerate(target_ids):
            neg_ids.append([])
            for j in range(self.neg_samples):
                n_id = self.vocab.neg_table[
                    random.choice(range(len(self.vocab.neg_table)))
                ]
                while n_id == target_ids[i] or n_id == context_ids[i]:
                    n_id = self.vocab.neg_table[
                        random.choice(range(len(self.vocab.neg_table)))
                    ]
                neg_ids[i].append(n_id)

        # total_neg_samples = len(target_ids) * self.neg_samples
        # neg_ids = self.vocab.neg_table[self.neg_index: self.neg_index + total_neg_samples]
        # self.neg_index = (self.neg_index + total_neg_samples) % len(self.vocab.neg_table)
        #
        # if len(neg_ids) != total_neg_samples:
        #     neg_ids = np.concatenate((neg_ids, self.vocab.neg_table[0: self.neg_index]))
        # neg_ids = torch.from_numpy(neg_ids).view(len(target_ids), -1)
        ###################################################################################################

        neg_ids = torch.Tensor(np.array(neg_ids)).long()
        return target_ids, context_ids, neg_ids

    def collate_fn(self, batch_list):
        target_ids = [target_id for batch in batch_list for target_id in batch[0]]
        context_ids = [context_id for batch in batch_list for context_id in batch[1]]
        neg_ids = [neg_samples for batch in batch_list for neg_samples in batch[2]]

        if len(target_ids) == 0:
            return None, None, None

        target_ids = torch.LongTensor(target_ids)
        context_ids = torch.LongTensor(context_ids)
        neg_ids = torch.stack(neg_ids)

        return target_ids, context_ids, neg_ids


def similarity(word1_dm, word2_dm):
    # trace inner product
    # Efficient way to compute trace of matrix product
    trace = (word1_dm * word2_dm.T).sum().item()
    # Normalise
    trace = trace / (
            math.sqrt((word1_dm ** 2).sum()) * math.sqrt((word2_dm ** 2).sum())
    )
    return trace
