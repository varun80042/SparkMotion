import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn

from tools.data_utils import read_csv, convert_to_one_hot, read_glove_vecs, sentences_to_indices
from tools.model import NN, pretrained_embedding_layer
from tools.train import train, evaluate
from tools.predict import predict

checkpoint_dir = './checkpoint/'
os.makedirs(checkpoint_dir, exist_ok=True)

X_train, Y_train = read_csv('./data/train.csv')
X_test, Y_test = read_csv('./data/test.csv')

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('./data/glove.6B.200d.txt')

maxLen = len(max(X_train, key=len).split())
X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)

Y_train = torch.tensor(Y_train).type(torch.LongTensor)
Y_test = torch.tensor(Y_test).type(torch.LongTensor)

embedding, vocab_size, embedding_dim = pretrained_embedding_layer(word_to_vec_map, word_to_index, non_trainable=True)

hidden_dim = 128
output_size = 5
model = NN(embedding, embedding_dim, hidden_dim, vocab_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)

batch_size = 32
train_dataset = TensorDataset(torch.tensor(X_train_indices).type(torch.LongTensor), Y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(torch.tensor(X_test_indices).type(torch.LongTensor), Y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 50
train(model, train_loader, test_loader, criterion, optimizer, epochs, device)

test_loss, accuracy = evaluate(model, test_loader, criterion, device)
print("Test Loss: {:.3f}.. ".format(test_loss), "Test Accuracy: {:.3f}".format(accuracy))

torch.save(model, checkpoint_dir + 'model.pth')
