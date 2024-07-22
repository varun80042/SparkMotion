import os
import boto3

def getModel():
    S3_MODEL = 's3://sparkmotion/model.pth'
    CHECKPOINT_DIR = './checkpoint/'
    LOCAL_MODEL_PATH = CHECKPOINT_DIR + 'model.pth'

    def get(s3_path, local_path):
        s3 = boto3.client('s3')
        bucket_name, key_name = s3_path.replace("s3://", "").split("/", 1)
        s3.download_file(bucket_name, key_name, local_path)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if not os.path.exists(LOCAL_MODEL_PATH):
        print("Fetching the model from AWS...")
        get(S3_MODEL, LOCAL_MODEL_PATH)