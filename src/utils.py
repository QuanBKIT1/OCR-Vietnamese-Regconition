import gdown
import zipfile
import os
import numpy as np 
from itertools import groupby

def download_dataset():
    # Check if dataset is available
    folder_path = './Handwritten OCR'
    if os.path.exists(folder_path):
        print(f"The folder '{folder_path}' exists.")
        # Check number of files
        print('Found ', len(os.listdir('./Handwritten OCR/training_data/new_train')), ' items in training dir')
        print('Found ', len(os.listdir('./Handwritten OCR/public_test_data/new_public_test')), ' items in public test dir')
        return
    folder_link = "https://drive.google.com/drive/folders/1dlhSYYrLE0GMUOUV-GDmNcJs2_Tu4KYa?usp=drive_link"
    gdown.download_folder(folder_link)
    zip_file_paths = ['./Handwritten OCR/training_data.zip', './Handwritten OCR/public_test_data.zip']
    extract_dirs = ['./Handwritten OCR/training_data', './Handwritten OCR/public_test_data']

    for zip_file_path, extract_dir in zip(zip_file_paths, extract_dirs):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    print('Download successfully')

def  encode_text(text, vocab):
    return [vocab.index(i) for i in text]


def ctc_decoder(predictions, chars):
    """ CTC greedy decoder for predictions

    Args:
        predictions (np.ndarray): predictions from model
        chars (typing.Union[str, list]): list of characters
    Returns:
        typing.List[str]: list of words
    """
    # use argmax to find the index of the highest probability
    argmax_preds = np.argmax(predictions, axis=-1)
    # use groupby to find continuous same indexes
    grouped_preds = [[k for k,_ in groupby(preds)] for preds in argmax_preds]

    # convert indexes to chars
    texts = ["".join([chars[k] for k in group if k < len(chars)]) for group in grouped_preds]

    return texts
