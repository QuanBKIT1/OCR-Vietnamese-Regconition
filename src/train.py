import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from utils import download_dataset
from loader import DataGenerator
from configs import *
from model import construct_model
from loss import CTCloss
from metric import CWERMetric

def train():
    # Download data
    download_dataset()

    # ----------- Preprocessing data -----------

    # Get filepath and label
    dataset = []

    words = open('./Handwritten OCR/train_gt.txt', "r", encoding='utf-8').readlines()
    for line in words:
        line_split = line.split("\t")
        label = line_split[-1].rstrip("\n")
        rel_path = line_split[0]
        dataset.append([rel_path, label])

    number_train = int(len(dataset) * TRAIN_VAL_SPLIT)
    train_list_IDs = [item[0] for item in dataset[:number_train]]
    train_labels = [item[1] for item in dataset[:number_train]]
    val_list_IDs = [item[0] for item in dataset[number_train:]]
    val_labels = [item[1] for item in dataset[number_train:]]

    train_generator = DataGenerator(train_list_IDs, train_labels)
    val_generator = DataGenerator(val_list_IDs, val_labels)

    # ----------- Preprocessing data -----------

    # Construct model
    model = construct_model(input_dim=(HEIGHT, WIDTH, 1),
                            output_dim=len(VOCAB))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=CTCloss(),
        metrics=[CWERMetric(padding_token=len(VOCAB))],)

    # Define callback
    earlystopper = EarlyStopping(monitor="val_CER", patience=10, verbose=1)
    checkpoint = ModelCheckpoint(filepath='../../../checkpoint',
                                 monitor="val_CER",
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='min',
                                 verbose=1)
    os.chdir(os.path.join(os.getcwd(), TRAINING_DIR))

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=TRAIN_EPOCHS,
        callbacks=[earlystopper, checkpoint],
    )
    os.chdir('../../..')
    return model

if __name__ == '__main__':
    model = train()
    model.save('./model/ocr_model.h5')
