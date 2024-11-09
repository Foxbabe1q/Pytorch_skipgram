import io
import os
import sys
import requests
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
def load_data():
    with open('fil9','r') as f:
        data = f.read()
        # print(data[:100])[:10000000]
        corpus = data.split()[:2000000]
        # print(corpus[:100])[:10000000]
        print('corpus_size: ', len(corpus))
        return corpus

def build_word_freq_tuple(corpus):
    word_freq_dict = {}
    for word in corpus:
        if word in word_freq_dict:
            word_freq_dict[word] += 1
        elif word not in word_freq_dict:
            word_freq_dict[word] = 1
    word_freq_tuple = sorted(word_freq_dict.items(), key=lambda x: x[1], reverse=True)
    # print(word_freq_tuple[:10])
    return word_freq_tuple

def build_word_id_dict(corpus):
    word_freq_dict = {}
    word_freq_tuple = build_word_freq_tuple(corpus)
    for word, freq in word_freq_tuple:
        word_freq_dict[word] = freq

    word_id_dict = {}
    id_word_dict = {}

    for index, (word, freq) in enumerate(word_freq_tuple):
        word_id_dict[word] = index
        id_word_dict[index] = word

    print('vocabulary size: ', len(word_id_dict))

    # for _, (word, id) in zip(range(20), word_id_dict.items()):
    #     print('word: ',word, 'id: ', id, 'freq: ', word_freq_dict[word])

    return word_freq_dict, word_id_dict, id_word_dict


def convert_corpus_id(corpus, word_id_dict):
    id_corpus = []
    for word in corpus:
        id_corpus.append(word_id_dict[word])

    # print(id_corpus[:20])

    return id_corpus


def subsampling(corpus, word_freq_dict):
    corpus = [word for word in corpus if not np.random.rand() < (1 - (np.sqrt(1e-5 * len(corpus) / word_freq_dict[word])))]
    print('corpus_size after subsampling: ', len(corpus))
    return corpus


def build_negative_sampling_dataset(corpus, word_id_dict, id_word_dict, negative_sample_size = 10, max_window_size = 3):
    dataset = []
    for center_word_idx, center_word in enumerate(corpus):
        window_size = np.random.randint(1, max_window_size+1)
        positive_range = (max(0, center_word_idx - window_size), min(len(corpus) - 1, center_word_idx + window_size))
        positive_samples = [corpus[word_idx] for word_idx in range(positive_range[0], positive_range[1]+1) if word_idx != center_word_idx]

        for positive_sample in positive_samples:
            dataset.append((center_word, positive_sample, 1))

        sample_idx_list = np.arange(len(word_id_dict))
        j = corpus[positive_range[0]: positive_range[1]+1]
        sample_idx_list = np.delete(sample_idx_list, j)
        negative_samples = np.random.choice(sample_idx_list, size=negative_sample_size, replace=False)
        for negative_sample in negative_samples:
            dataset.append((center_word, negative_sample, 0))

    print('20 samples of the dataset')
    for i in range(20):
        print('center_word:', id_word_dict[dataset[i][0]], 'target_word:', id_word_dict[dataset[i][1]], 'label',
              dataset[i][2])
    return dataset


class create_dataset(Dataset):
    def __init__(self, dataset):
        self.center_idx = [x[0] for x in dataset]
        self.target_idx = [x[1] for x in dataset]
        self.label = [x[2] for x in dataset]

    def __len__(self):
        return len(self.center_idx)

    def __getitem__(self, idx):
        return (torch.tensor(self.center_idx[idx], dtype=torch.int64, device=device), torch.tensor(self.target_idx[idx], dtype=torch.int64, device=device),
                torch.tensor(self.label[idx], dtype=torch.float32, device=device, requires_grad=True))


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(SkipGram, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.out_embedding = nn.Embedding(self.vocab_size, self.embedding_size)

        init_range = (1 / embedding_size) ** 0.5
        nn.init.uniform_(self.embedding.weight, -init_range, init_range)
        nn.init.uniform_(self.out_embedding.weight, -init_range, init_range)

    def forward(self, center_idx, target_idx, label):
        center_embedding = self.embedding(center_idx)
        target_embedding = self.embedding(target_idx)
        sim = torch.mul(center_embedding, target_embedding)
        sim = torch.sum(sim, dim=1, keepdim=False)
        loss = F.binary_cross_entropy_with_logits(sim, label,reduction='sum')
        return loss

def train(vocab_size, dataset):
    my_skipgram = SkipGram(vocab_size = vocab_size, embedding_size=300)
    my_skipgram.to(device)
    my_dataset = create_dataset(dataset)
    my_dataloader = DataLoader(my_dataset, batch_size=64, shuffle=True)
    optimizer = optim.Adam(my_skipgram.parameters(), lr=0.001)
    epochs = 10
    loss_list = []
    start_time = time.time()

    for epoch in range(epochs):
        total_loss = 0
        total_sample = 0
        for center_idx, target_idx, label in my_dataloader:
            loss = my_skipgram(center_idx, target_idx, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_sample += len(center_idx)
        print(f'epoch: {epoch+1}, loss = {total_loss/total_sample}, time = {time.time() - start_time : .2f}')
        loss_list.append(total_loss/total_sample)
    plt.plot(np.arange(1, epochs + 1),loss_list)
    plt.title('Loss_curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(np.arange(1, epochs + 1))
    plt.savefig('loss_curve.png')
    plt.show()
    torch.save(my_skipgram.state_dict(), 'skip_gram.pt')

def predict(word, vocab_size, word_id_dict):
    if word not in word_id_dict:
        print(f"Word '{word}' not found in the vocabulary.")
        return None
    my_skipgram = SkipGram(vocab_size = vocab_size, embedding_size=300)
    my_skipgram.load_state_dict(torch.load('skip_gram.pt'))
    my_skipgram.to(device)
    my_skipgram.eval()
    word_id = torch.tensor(word_id_dict[word], device=device, dtype=torch.int64)
    print(
        f"Predicting the embedding vector for word '{word}':\n{my_skipgram.embedding(word_id)}"
    )

def similarity(word, vocab_size, word_id_dict, id_word_dict, neighbors = 5):
    if word not in word_id_dict:
        print(f"Word '{word}' not found in the vocabulary.")
        return None
    my_skipgram = SkipGram(vocab_size=vocab_size, embedding_size=300)
    my_skipgram.load_state_dict(torch.load('skip_gram.pt', weights_only=True))
    my_skipgram.to(device)
    my_skipgram.eval()
    word_id = torch.tensor(word_id_dict[word], device=device, dtype=torch.int64)
    word_embedding = my_skipgram.embedding(word_id)
    similarity_score = {}
    for idx in word_id_dict.values():
        other_word_embedding = my_skipgram.embedding(torch.tensor(idx, device=device, dtype=torch.int64))
        sim = torch.matmul(word_embedding, other_word_embedding)/(torch.norm(word_embedding, dim=0, keepdim=False) * torch.norm(other_word_embedding, dim=0, keepdim=False))
        similarity_score[id_word_dict[idx]] = sim.item()
    nearest_neighbors = sorted(similarity_score.items(), key=lambda x: x[1], reverse=True)[:5]
    print(nearest_neighbors)
    return nearest_neighbors


if __name__ == '__main__':
    corpus = load_data()
    word_freq_dict, word_id_dict, id_word_dict = build_word_id_dict(corpus)
    corpus = subsampling(corpus, word_freq_dict)
    corpus = convert_corpus_id(corpus, word_id_dict)
    # dataset = build_negative_sampling_dataset(corpus, word_id_dict, id_word_dict, negative_sample_size = 10)
    # train(len(word_id_dict), dataset)
    predict('sport', len(word_id_dict), word_id_dict)
    similarity('sport', len(word_id_dict), word_id_dict, id_word_dict, neighbors = 5)
# epoch: 1, loss = 1.1638301322539766, time =  358.33
# epoch: 2, loss = 1.423060077407835, time =  716.48
# epoch: 3, loss = 3.5354185104784226, time =  1074.58
# epoch: 4, loss = 13.870041244686394, time =  1433.45
# epoch: 5, loss = 39.63488978511493, time =  1792.24