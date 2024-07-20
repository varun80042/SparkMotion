import os
import sys
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from tools.data_utils import read_glove_vecs, sentences_to_indices
from tools.model import NN, pretrained_embedding_layer
from tools.predict import predict

glove_file = './data/glove.6B.200d.txt'
model_file = './checkpoint/model.pth'
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(glove_file)
maxLen = 10  

model = torch.load(model_file)
model.eval()

labels_dict = {
    0 : "Loving",
    1 : "Playful",
    2 : "Happy",
    3 : "Annoyed",
    4 : "Foodie",
}

def process_rdd(rdd):
    if not rdd.isEmpty():
        sentences = rdd.collect()
        for sentence in sentences:
            predict(model, sentence, word_to_index, maxLen, labels_dict)

if __name__ == "__main__":
    sc = SparkContext(appName="SparkMotion")
    ssc = StreamingContext(sc, 10)
    ssc.sparkContext.setLogLevel("WARN")

    lines = ssc.socketTextStream("localhost", 9999)
    lines.foreachRDD(process_rdd)

    ssc.start()
    ssc.awaitTermination()
