import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NN(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, vocab_size, output_dim):
        super(NN, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = embedding
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, dropout=0.5, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, sentence):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sentence = sentence.to(device)
        embeds = self.word_embeddings(sentence)
        h0 = torch.zeros(2, sentence.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(2, sentence.size(0), self.hidden_dim).to(device)
        self.lstm.flatten_parameters()
        lstm_out, h = self.lstm(embeds, (h0, c0))
        lstm_out = lstm_out[:, -1, :]
        lstm_out = F.dropout(lstm_out, 0.5)
        fc_out = self.fc(lstm_out)
        out = F.softmax(fc_out, dim=1)
        return out

def pretrained_embedding_layer(word_to_vec_map, word_to_index, non_trainable=True):
    num_embeddings = len(word_to_index) + 1
    embedding_dim = word_to_vec_map["cucumber"].shape[0]
    weights_matrix = np.zeros((num_embeddings, embedding_dim))
    for word, index in word_to_index.items():
        weights_matrix[index, :] = word_to_vec_map[word]
    embed = nn.Embedding.from_pretrained(torch.from_numpy(weights_matrix).type(torch.FloatTensor), freeze=non_trainable)
    return embed, num_embeddings, embedding_dim
