import torch
import matplotlib.pyplot as plt
import numpy as np

def train(model, train_loader, test_loader, criterion, optimizer, epochs=10, device='cpu'):
    model.to(device)
    train_losses, test_losses, accuracies = [], [], []

    for e in range(epochs):
        model.train()
        running_loss = 0
        
        for sentences, labels in train_loader:
            sentences, labels = sentences.to(device), labels.to(device)
            optimizer.zero_grad()
            pred = model(sentences)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        model.eval()
        test_loss = 0
        accuracy = 0
        with torch.no_grad():
            for sentences, labels in test_loader:
                sentences, labels = sentences.to(device), labels.to(device)
                log_ps = model(sentences)
                test_loss += criterion(log_ps, labels).item()
                
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        train_losses.append(running_loss / len(train_loader))
        test_losses.append(test_loss / len(test_loader))
        accuracies.append(accuracy / len(test_loader) * 100)

        print(f"Epoch: {e+1}/{epochs}.. "
              f"Training Loss: {running_loss / len(train_loader):.3f}.. "
              f"Test Loss: {test_loss / len(test_loader):.3f}.. "
              f"Test Accuracy: {accuracy / len(test_loader):.3f}%")

    plt.figure(figsize=(20, 5))
    plt.plot(train_losses, c='b', label='Training loss')
    plt.plot(test_losses, c='r', label='Testing loss')
    plt.xticks(np.arange(0, epochs))
    plt.title('Losses')
    plt.legend(loc='upper right')
    plt.show()

    plt.figure(figsize=(20, 5))
    plt.plot(accuracies)
    plt.xticks(np.arange(0, epochs))
    plt.title('Accuracy')
    plt.show()

def evaluate(model, test_loader, criterion, device='cpu'):
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for sentences, labels in test_loader:
            sentences, labels = sentences.to(device), labels.to(device)
            ps = model(sentences)
            test_loss += criterion(ps, labels).item()

            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
    model.train()
    return test_loss / len(test_loader), accuracy / len(test_loader)
