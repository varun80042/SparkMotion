import torch
import numpy as np
from tools.data_utils import sentences_to_indices

def predict(model, input_text, word_to_index, max_len, labels_dict):
    x_test = np.array([input_text])
    X_test_indices = sentences_to_indices(x_test, word_to_index, max_len)
    sentences = torch.tensor(X_test_indices).type(torch.LongTensor)
    model.eval()
    with torch.no_grad():
        ps = model(sentences)
    top_p, top_class = ps.topk(1, dim=1)
    label = int(top_class[0][0])
    print(f"Input Text: {input_text}\nEmotion: {labels_dict[label]}")
    return label
