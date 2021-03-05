import torch
from model import LSTM_model
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np


def load_model(model_path):
    with open(model_path, 'rb') as f:
        return torch.load(f, map_location=torch.device('cpu'))


def test(args, model, dataset):
    model.eval()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    total_correct = 0
    total_incorrect = 0
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

        preds = torch.argmax(output, dim=1)
        if sum(labels) > 0:
            fscores.append(f1_score(labels, preds))
        correct = sum((preds ^ labels) == 0).item()
        total_correct += correct
        total_incorrect += (args.batch_size - correct)

        total_pos_samples += torch.sum(labels)
        total_pos_out += confusion_matrix(labels, torch.argmax(output, dim=1), labels=[0, 1]).ravel()[-1]
    results = {}
    results['TPR'] = total_pos_out / total_pos_samples
    results['total_correct'] = total_correct
    results['total_incorrect'] = total_incorrect
    results['accuracy'] = total_correct / (total_incorrect + total_correct)
    results['avg_f1'] = np.mean(fscores)
    print("TEST\t" + "\t".join([f"{k}= {v}" for k, v in results.items()]))
    return
