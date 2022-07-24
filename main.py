from tqdm import tqdm

from utils.vocab import Vocab
from utils.Dataset import skip_gram_dataset
from model.skip_gram import skip_gram_model

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import random
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    random.seed(seed)

    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True


set_seed(924)
epoch = 20
train_batch_size = 4
test_batch_size = 1
lr = 0.0001


def collate_fn(samples):
    inputs = torch.tensor([sample["input"] for sample in samples])
    labels = torch.tensor([sample["label"] for sample in samples])

    return inputs, labels


# Vocab.from_data_get_vocab("data/train/train.zh","data/vocab/vocab.zh")

vocab = Vocab.load_vocab("data/vocab/vocab.zh")
# 加载句子
data = skip_gram_dataset.load_data("data/train/train.zh")[:64]
# 处理为skip_gram格式
data = skip_gram_dataset.skip_gram(data, vocab, window_size=2)
# 划分数据
train_data, dev_data, test_data = skip_gram_dataset.split_data(data, dev_num=0, test_num=100)
train_dataset = skip_gram_dataset(data=train_data)
test_dataset = skip_gram_dataset(data=test_data)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)

model = skip_gram_model(vocab)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

loss_func = torch.nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=lr)
model.train()
for i in range(epoch):
    total_loss = 0
    acc = 0
    for batch in train_dataloader:
        inputs = batch[0].to(device)
        labels = batch[1]
        output = model(inputs)
        optimizer.zero_grad()
        loss = loss_func(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        acc += (torch.argmax(output, dim=-1) == labels).sum().item()

    print(f"epoch:{i} batch_loss:{total_loss / len(train_dataloader)} acc:{acc / (len(train_dataloader) * 4)}")

test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=True, collate_fn=collate_fn)
model.eval()
total_loss = 0
acc = 0
target = {1: '是', 0: "否"}
for batch in test_dataloader:
    inputs = batch[0].to(device)
    labels = batch[1]
    output = model(inputs)
    loss = loss_func(output, labels)
    total_loss += loss.item()
    acc += (torch.argmax(output, dim=-1) == labels).sum().item()
    output = torch.argmax(output, dim=-1)
    print(
        f"是否共现：{vocab.id2token[inputs[0][0].item()]} {vocab.id2token[inputs[0][1].item()]} 答案：{target[labels[0].item()]} 预测：{target[output.item()]}")

print(f"loss:{total_loss / len(test_dataloader)} acc:{acc / (len(test_dataloader))}")
