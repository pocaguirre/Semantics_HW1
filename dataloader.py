###########
# pytonNSS
# author: rewicks
# edited by: pocaguirre
###########

import random
import torch

torch.manual_seed(42)
random.seed(42)


class Vocabulary():
    def __init__(self, itos=['<unk>', '<pad>'], stoi={'<unk>':0, '<pad>':1}):
        self.itos = itos
        self.stoi = stoi

    def add_word(self, token):
        if token not in self.stoi:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)

    def tok2ind(self, token):
        # is there no UNK in model?
        return self.stoi.get(token, self.stoi['<unk>'])

    def ind2tok(self, ind):
        return self.itos[ind]

    def convert_string(self, string):
        output = []
        for s in string:
            output.append(self.tok2ind(s))
        return output

    def convert_indices(self, ind_list):
        output = []
        for i in ind_list:
            output.append(self.ind2tok(i))
        return output


class Dataset():
    def __init__(self, dataset_path, batch_size, vocab=None, target_label='AGENT'):
        if vocab:
            self.vocab = vocab
            self.new_vocab = False
        else:
            self.vocab = Vocabulary()
            self.new_vocab = True
        self.target_label = target_label
        data, max_length = self.process(dataset_path)
        data, data_len, labels = self.batchify(data, batch_size, self.vocab.tok2ind("<pad>"), max_length)
        self.data = data
        self.data_len = data_len
        self.labels = labels
        self.batch_size = batch_size

    def process_sentence(self, tokens):
        if self.new_vocab:
            for token in tokens:
                self.vocab.add_word(token)

    def process(self, dataset_path):
        data = []
        longest_line = 0
        with open(dataset_path) as inputfile:
            last_line = None
            for i, line in enumerate(inputfile):
                if i == 0:
                    continue
                #line = self.tokenizer.encode(line)
                line = line.split("\t")
                self.process_sentence(line[3].split())
                tokenized = self.vocab.convert_string(line[3].split())
                if longest_line > len(tokenized):
                    longest_line = len(tokenized)
                if line[4][:-1] == self.target_label:
                    label = 1
                else:
                    label = 0
                data.append(((line[1], line[2], tokenized), label))
        return data, longest_line

    def batchify(self, data, batch_size, pad_index, max_length, seed=14):
        random.seed(seed)
        random.shuffle(data)

        data_batches = []
        data_len_batches = []
        label_batches = []
        data_batch = []
        data_len_batch = []
        label_batch = []
        for example, label in data:
            if len(data_batch) == batch_size:
                data_batches.append(torch.nn.utils.rnn.pad_sequence(data_batch, padding_value=pad_index).transpose(0,1))
                data_len_batches.append(torch.tensor(data_len_batch))
                label_batches.append(torch.tensor(label_batch))
                #label_batches.append(torch.tensor(label_batch))
                data_batch = []
                label_batch = []
                data_len_batch = []
            data_batch.append(torch.tensor([int(example[0]), int(example[1])] + example[2]))
            data_len_batch.append(len(example[2])-1)
            label_batch.append(torch.tensor(label))

            #label_batch.append(label + [ [0 for x in range(len(label[0]))] for y in range(len(example), max_length)])
        return data_batches, data_len_batches, label_batches

    def get_batches(self):
        for data, data_len, label in zip(self.data, self.data_len, self.labels):
            yield data, data_len, label