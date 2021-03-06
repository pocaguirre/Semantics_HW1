from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import argparse
from dataloader import Dataset
from model import LSTM_model
from main import evaluate
import math
import operator


def parse_arguments():
    """
    Parse command-line to identify configuration filepath.

    Args:
        None

    Returns:
        args (argparse Object): Command-line argument holder.
    """
    # Initialize Parser Object
    parser = argparse.ArgumentParser(description="Run modeling experiments")

    parser.add_argument("--gpu",
                        action="store_true",
                        help='use gpu')
    parser.add_argument("--train_data_path",
                        type=str,
                        default='dataset_train.tsv',
                        help='Path to train dataset')
    parser.add_argument("--dev_data_path",
                        type=str,
                        default="dataset_dev.tsv",
                        help="Path to dev dataset")
    parser.add_argument("--test_data_path",
                        type=str,
                        default='dataset_test.tsv',
                        help='Path to test dataset')
    parser.add_argument("--epoch",
                        type=int,
                        default=20,
                        help="epochs")
    parser.add_argument("--target_label",
                        type=str,
                        default='AGENT',
                        help='model prefix')
    # Other args
    parser.add_argument("--random_state", type=int, default=42)

    # Parse Arguments
    args = parser.parse_args()
    return args


class Tune(object):
    def __init__(self, config, best, train, args):
        self.config = config
        self.results = {k: {c: None for c in v} for k, v in config.items()}
        self.train = train
        self.best = best
        self.args = args

    def run(self):
        for k, v in self.results.items():
            for val in v.keys():
                tmp = self.best.copy()
                tmp[k] = val
                self.train(tmp, self.args, k, val)
            best_val = max(self.results[k], key=lambda key: self.results[k][key]['f1'])
            self.best[k] = best_val
            print("CURRENT BEST")
            print(self.best)
        return self.best, self.results

    def report(self, k, v, **kwargs):
        if self.results[k][v]:
            if self.results[k][v]['loss'] > kwargs['loss']:
                self.results[k][v]['loss'] = kwargs['loss']
            if self.results[k][v]['f1'] < kwargs['f1']:
                self.results[k][v]['f1'] = kwargs['f1']
        else:
            self.results[k][v] = kwargs


def train(config, args, k, v):
    print("Loading datasets...")
    dataset = Dataset(args.train_data_path, config['batch_size'], target_label=args.target_label)
    valid_dataset = Dataset(args.dev_data_path, config['batch_size'], vocab=dataset.vocab, target_label=args.target_label)
    net = LSTM_model(config['embedding_dim'], config['hidden_dim'], len(dataset.vocab.itos), config['hidden_dim2'])
    print(config)

    if args.gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    net.to(device)
    weight_tensor = torch.tensor([1.0, 1.0*config['weight']]).to(device)
    criterion = nn.CrossEntropyLoss(weight_tensor)
    params = net.parameters()
    optimizer = torch.optim.Adam(params, lr=config['lr'])

    best_loss = math.inf
    best_f1 = -math.inf

    for ep in range(args.epoch):
        total_loss = 0
        processed_batches = 0
        for data, data_len, labels in dataset.get_batches():
            tokens = data[:, 2:]
            preds_args = data[:, 0:2]
            tokens = tokens.to(device)
            labels = labels.to(device)
            data_lens = data_len.to(device)
            preds_args = preds_args.to(device)
            # Step 3. Run our forward pass.
            # tag_scores = model(sentence_in)
            # import pdb;pdb.set_trace()
            output = net(preds_args, tokens, data_lens)

            loss = criterion(output, labels)
            total_loss += loss
            net.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 0.5)
            optimizer.step()
            processed_batches += 1

            # print statistics
            total_loss += loss.item()
            if processed_batches % 20 == 0:  # print every 20 mini-batches
                print("[%d, %5d] loss: %.3f" % (ep + 1, processed_batches, total_loss))

            if processed_batches % 100 == 0:
                result = evaluate(args, net, criterion, valid_dataset)
                net.train()
                if result['total_loss'] < best_loss:
                    best_loss = result['total_loss']
                if result['avg_f1'] > best_f1:
                    best_f1 = result['avg_f1']
        tune.report(k, v, loss=best_loss, f1=best_f1)
    result = evaluate(args, net, criterion, valid_dataset)
    net.train()
    if result['total_loss'] < best_loss:
        best_loss = result['total_loss']
    if result['avg_f1'] > best_f1:
        best_f1 = result['avg_f1']
    tune.report(k, v, loss=best_loss, f1=best_f1)
    print("Finished Training")


def main(args):
    print(f"LABEL: {args.target_label}")
    config = {
        "lr": np.linspace(1e-4, 1e-1, num=5),
        "batch_size": [25, 50, 75],
        "embedding_dim": [25, 50, 75, 100, 200],
        "hidden_dim": [50, 100, 200, 300, 500],
        "hidden_dim2": [0, 50, 100, 200],
        "weight": [1, 10, 20, 50]
    }
    best = {
        "lr": 0.01,
        "batch_size": 25,
        "embedding_dim": 50,
        "hidden_dim": 200,
        "hidden_dim2": 0,
        "weight": 1
    }
    global tune
    tune = Tune(config, best, train, args)
    best, result = tune.run()
    print("BEST")
    print(best)
    print("ALL RESULTS")
    print(result)


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    args = parse_arguments()
    main(args)