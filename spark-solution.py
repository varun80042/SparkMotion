import os
import sys
import torch
from pyspark import SparkContext
from pyspark.streaming import StreamingContext

from tools.model import NN, pretrained_embedding_layer
from tools.predict import predict
from tools.get_model import getModel

LOCAL_MODEL_PATH = './checkpoint/model.pth'
MAX_LEN = 20  # max number of words

if not os.path.exists(LOCAL_MODEL_PATH):
    print("No checkpoint found!")
    getModel()

model = torch.load(LOCAL_MODEL_PATH)
model.eval()

def process_rdd(rdd):
    if not rdd.isEmpty():
        sentences = rdd.collect()
        for sentence in sentences:
            predict(model, sentence, MAX_LEN)

if __name__ == "__main__":
    sc = SparkContext(appName="SparkMotion")
    ssc = StreamingContext(sc, 10)
    ssc.sparkContext.setLogLevel("ERROR")

    lines = ssc.socketTextStream("localhost", 9999)
    lines.foreachRDD(process_rdd)

    ssc.start()
    ssc.awaitTermination()
