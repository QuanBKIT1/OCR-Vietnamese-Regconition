import cv2
from configs import *
import numpy as np
from utils import ctc_decoder
from model import construct_model
import os
from tqdm import tqdm 
import tensorflow as tf

def predict(model, image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (WIDTH, HEIGHT))
    image_pred = np.expand_dims(image, axis=0).astype(np.float32)
    preds = model.predict(image_pred, verbose=0)
    text = ctc_decoder(preds, VOCAB)
    return text


def make_predict(model, directory):
    os.chdir(os.path.join(os.getcwd(), directory))
    # Code to predict public test data
    image_paths = sorted(os.listdir())
    predict_data = []
    for image_path in tqdm(image_paths):
        text = predict(model,image_path)[0]
        predict_data.append((image_path,text))
        
    os.chdir('../../..')
    return predict_data

if __name__ == '__main__':
    # Load model
    model = construct_model(input_dim=(HEIGHT, WIDTH, 1),
                            output_dim=len(VOCAB))
    # model.load_weights('./checkpoint/cp.ckpt')
    model = tf.keras.models.load_model('./model/ocr_model.h5')

    predict_data = make_predict(model, PUBLIC_TEST_DIR)

    # Fix empty string
    for i in range(len(predict_data)):
        if predict_data[i][1] == "":
            predict_data[i] = (predict_data[i][0],'a')

    with open('./prediction.txt', 'w') as file:
        for data in predict_data:
            dump = f'{data[0]} {data[1]}\n'
            file.write(dump)

    # print(f'./{PUBLIC_TEST_DIR}/public_test_img_0')
    # text = predict(model,f'./{PUBLIC_TEST_DIR}/public_test_img_0.jpg')[0]
    # print(text)
