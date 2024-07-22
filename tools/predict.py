import torch
import numpy as np

from tools.data_utils import sentences_to_indices, read_glove_vecs

GLOVE_FILE = './data/glove.6B.200d.txt'
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(GLOVE_FILE)

labels_dict = {
    0 : "Loving",
    1 : "Playful",
    2 : "Happy",
    3 : "Annoyed",
    4 : "Foodie",
}

def predict(model, input_text, max_len):
    words = input_text.split()
    if len(words) < 2:
        print(f"Input Text: {input_text}\nEmotion: NA")
    else:
        x_test = np.array([input_text])
        X_test_indices = sentences_to_indices(x_test, word_to_index, max_len)
        sentences = torch.tensor(X_test_indices).type(torch.LongTensor)
        model.eval()
        with torch.no_grad():
            ps = model(sentences)
        top_p, top_class = ps.topk(1, dim=1)
        label = int(top_class[0][0])
        print(f"Input Text: {input_text}\nEmotion: {labels_dict[label]}")
