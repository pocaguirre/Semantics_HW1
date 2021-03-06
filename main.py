import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader import Dataset
import argparse
import math
from model import LSTM_model
import time
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from tester import load_model, test


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

    parser.add_argument("--mode",
                        type=str,
                        default='train',
                        help='either train or test')
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

    # training args
    parser.add_argument("--batch_size",
                        type=int,
                        default=25,
                        help="batch size by examples")
    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.001,
                        help="learning rate")
    parser.add_argument("--epoch",
                        type=int,
                        default=25,
                        help="epochs")
    parser.add_argument("--embedding_dim",
                        type=int,
                        default=50,
                        help="embedding dimmension")
    parser.add_argument("--hidden_dim",
                        type=int,
                        default=200,
                        help="hidden dimension")
    parser.add_argument("--hidden_dim2",
                        type=int,
                        default=0,
                        help="hidden dimension 2")
    parser.add_argument("--output_path_prefix",
                        type=str,
                        default='agent',
                        help='model prefix')
    parser.add_argument("--target_label",
                        type=str,
                        default='AGENT',
                        help='model prefix')
    parser.add_argument("--weight",
                        type=float,
                        default=1.0,
                        help='Loss weight ratio')

    # Test args
    parser.add_argument("--model_path",
                        type=str,
                        default='agent.output.model.best',
                        help='Path to test dataset')

    # Other args
    parser.add_argument("--random_state", type=int, default=42)

    # Parse Arguments
    args = parser.parse_args()
    return args


def save(output_prefix, output_model, args=None, vocab=None, postfix=''):
    torch.save([output_model.state_dict(), args, vocab], f'{output_prefix}.output.model{postfix}')


def evaluate(args, model, criterion, dataset):
    model.eval()

    if args.gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    total_correct = 0
    total_incorrect = 0
    loss = 0
    fscores = []
    total_pos_samples = 0
    total_pos_out = 0
    print(f"-------------------Beginning evaluation-------------------")
    for data, data_len, labels in dataset.get_batches():
        tokens = data[:, 2:]
        preds_args = data[:, 0:2]
        tokens = tokens.to(device)
        labels = labels.to(device)
        data_lens = data_len.to(device)
        preds_args = preds_args.to(device)

        with torch.no_grad():
            output = model(preds_args, tokens, data_lens)

        preds = torch.argmax(torch.softmax(output, dim=1), dim=1)
        if sum(labels) > 0:
            fscores.append(f1_score(labels.cpu(), preds.cpu()))
        correct = sum((preds ^ labels) == 0).item()
        total_correct += correct
        total_incorrect += (dataset.batch_size - correct)
        loss += criterion(output, labels)

        total_pos_samples += torch.sum(labels)
        total_pos_out += confusion_matrix(labels.cpu(), torch.argmax(torch.softmax(output, dim=1), dim=1).cpu(), labels=[0,1]).ravel()[-1]
    results = {}
    results['total_loss'] = loss.item()
    results['TPR'] = total_pos_out/total_pos_samples
    results['total_correct'] = total_correct
    results['total_incorrect'] = total_incorrect
    results['accuracy'] = total_correct / (total_incorrect + total_correct)
    results['avg_f1'] = np.mean(fscores)
    print("VALID\t"+"\t".join([f"{k}= {v}" for k,v in results.items()]))
    return results


def train(args):
    dataset = Dataset(args.train_data_path, args.batch_size, target_label=args.target_label)
    valid_dataset = Dataset(args.dev_data_path, args.batch_size, target_label=args.target_label)
    criterion = nn.CrossEntropyLoss(torch.tensor([1.0, args.weight]))
    lr = args.learning_rate
    model = LSTM_model(args.embedding_dim, args.hidden_dim, len(dataset.vocab.itos), args.hidden_dim2)
    best_loss = math.inf
    best_TPR = -math.inf
    best_f1 = -math.inf

    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

    start = time.time()
    if args.gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = model.to(device)

    for ep in range(args.epoch):
        total_loss = 0
        processed_batches = 0
        total_pos_samples = 0
        total_pos_out = 0
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
            output = model(preds_args, tokens, data_lens)

            loss = criterion(output, labels)
            total_loss += loss
            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 0.5)
            optimizer.step()
            processed_batches += 1

            total_pos_samples += torch.sum(labels).item()
            total_pos_out += confusion_matrix(labels.cpu(), torch.argmax(torch.softmax(output, dim=1), dim=1).cpu(), labels=[0,1]).ravel()[-1]

            if processed_batches % 10 == 0:
                print(f"TRAIN\tLoss= {loss.item()}\tTPR= {total_pos_out/total_pos_samples}")
            if processed_batches % 100 == 0:
                result = evaluate(args, model, criterion, valid_dataset)
                model.train()
                if result['total_loss'] < best_loss:
                    best_loss = result['total_loss']
                    print(f"Saving best model with loss: {result['total_loss']}")
                    save(args.output_path_prefix, model, args, dataset.vocab, postfix='.best')
                if result['TPR'] > best_TPR:
                    best_TPR = result['TPR']
                    print(f"Saving best model with TPR: {result['TPR']}")
                    save(args.output_path_prefix, model, args, dataset.vocab, postfix='.TPR.best')
                if result['avg_f1'] > best_f1:
                    best_f1 = result['avg_f1']
                    print(f"Saving best model with loss: {result['avg_f1']}")
                    save(args.output_path_prefix, model, args, dataset.vocab, postfix='.f1.best')
        result = evaluate(args, model, criterion, valid_dataset)
        model.train()
        if result['total_loss'] < best_loss:
            best_loss = result['total_loss']
            print(f"Saving best model with loss: {result['total_loss']}")
            save(args.output_path_prefix, model, args, dataset.vocab, postfix='.best')
        if result['TPR'] > best_TPR:
            best_TPR = result['TPR']
            print(f"Saving best model with TPR: {result['TPR']}")
            save(args.output_path_prefix, model, args, dataset.vocab, postfix='.TPR.best')
        if result['avg_f1'] > best_f1:
            best_f1 = result['avg_f1']
            print(f"Saving best model with loss: {result['avg_f1']}")
            save(args.output_path_prefix, model, args, dataset.vocab, postfix='.f1.best')


if __name__ == '__main__':
    args = parse_arguments()

    if args.mode == 'train':
        train(args)
    if args.mode == 'test':
        saved = load_model(args.model_path)
        model = LSTM_model(saved[1].embedding_dim, saved[1].hidden_dim, len(saved[2].itos), saved[1].hidden_dim2)
        model.load_state_dict(saved[0])
        model.eval()
        dataset = Dataset(args.train_data_path, args.batch_size, target_label=args.target_label)
        datasets = [
            dataset,
            Dataset(args.dev_data_path, args.batch_size, vocab=dataset.vocab, target_label=args.target_label),
            Dataset(args.test_data_path, args.batch_size, vocab=dataset.vocab, target_label=args.target_label)
        ]
        test(args, model, datasets)