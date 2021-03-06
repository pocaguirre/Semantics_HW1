import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


class LSTM_model(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, hidden_dim2=0):
        super(LSTM_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        if self.hidden_dim2 == 0:
            # The linear layer that maps from hidden state space to tag space
            self.hidden2tag = nn.Linear(hidden_dim * 3, 2)
        else:
            self.hidden2tag = nn.Sequential(
                                    nn.Linear(hidden_dim * 3, hidden_dim2),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim2, 2),
                                )

    def forward(self, pred_args, sentences, sentence_lens):
        embeds = self.word_embeddings(sentences)
        embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, sentence_lens.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(embeds)
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        indices = torch.cat((pred_args, (sentence_lens-1).view(-1,1)), dim=1)
        selected = [lstm_out[x, i, :] for x, i in enumerate(indices)]
        selected = torch.stack(selected)
        selected = selected.view(selected.shape[0], -1)
        tag_space = self.hidden2tag(selected)
        # tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_space