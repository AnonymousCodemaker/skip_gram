import torch
import torch.nn as nn


class skip_gram_model(nn.Module):
    def __init__(self, vocab, hidden_dim=16):
        super(skip_gram_model, self).__init__()
        self.embedding = nn.Embedding(len(vocab), hidden_dim)
        self.linear1 = nn.Linear(2, 1)
        self.linear2 = nn.Linear(hidden_dim, 2)
        self.hidden = None

    def forward(self, input):
        input = self.embedding(input)
        self.hidden = input
        output = self.linear1(torch.transpose(input, dim0=1, dim1=2)).squeeze(dim=-1)
        output = self.linear2(output)
        output = torch.softmax(output, dim=-1)
        return output
