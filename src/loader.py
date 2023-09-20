from tensorflow import keras
import numpy as np
import cv2
import os
from configs import *
from utils import  encode_text

class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, list_IDs, labels):
      self.list_IDs = list_IDs
      self.labels = labels
      self.batch_size = BATCH_SIZE
      print(f"Found {len(list_IDs)} instances")
    
    def __len__(self):
        return len(self.list_IDs) // self.batch_size

    def __getitem__(self, index):
      batch_IDs = self.list_IDs[index * self.batch_size:(index + 1) * self.batch_size]
      batch_labels = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
      # Load and preprocess data for the current batch
      X,Y = self.__data_generation(batch_IDs, batch_labels)
      return X,Y


    def __data_generation(self, batch_IDs, batch_labels):

      X = np.ones([len(batch_IDs), HEIGHT, WIDTH, 1])
      Y = np.ones([len(batch_IDs), MAX_TEXT_LENGTH],dtype=int)
      for i, id in enumerate(batch_IDs):
        img = cv2.imread(id, cv2.IMREAD_GRAYSCALE)    # (h, w)
        img = cv2.resize(img, (WIDTH, HEIGHT))  # (h, w)
        img = np.expand_dims(img, -1)  # (h, w, 1)
        X[i] = img

      for i, text in enumerate(batch_labels):
        text_encode = encode_text(text, VOCAB)
        Y[i] = text_encode + [len(VOCAB) for _ in range(MAX_TEXT_LENGTH-len(text_encode))]
      return X,Y
