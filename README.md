# Vietnamese Handwritten Text Recognition

_SoICT Hackathon 2023 - Vietnamese Handwritten Text Recognition 🏃‍♂️‍➡️_

_One love, one future ❤️_

Team name: **TS1-AI**

## Introduction

**Vietnamese Handwritten Text Recognition (HTR)** is a challenging yet crucial task in the field of computer vision. The goal is to accurately convert handwritten text images into machine-readable text. However, the diversity of handwriting styles.

In recent years, deep learning-based approaches have demonstrated remarkable performance. Nevertheless, building a robust Vietnamese HTR system remains a complex problem due to the unique characteristics of the Vietnamese language.

This project aims to develop a deep learning model, bidirectional stacked LSTM using CNN features as input with CTC loss to perform robust word recognition. We conducted extensive experiments on a large-scale Vietnamese handwritten text dataset to evaluate the performance of our model.

## Dataset Preparation

Vietnamese Handwritten Text Dataset HomePage: [Dataset](https://aihub.ml/competitions/426)

- **Training data:** This is a real-world dataset with labels, used to train the model. It consists of 103,000 images (including 51,000 form images, 48,000 wild images, and 4,000 GAN-generated images).
- **Public test:** This is an unlabeled dataset used for preliminary evaluation. It consists of 33,000 images (including 17,000 form images and 16,000 wild images).

**Data sample:**

|                                                |                                                |
| ---------------------------------------------- | ---------------------------------------------- |
| ![Alt Text](./visualize/train_img_102992.jpg)  | ![Alt Text](./visualize/train_img_102993.jpg)  |
| ![Alt Text](./visualize/public_test_img_6.jpg) | ![Alt Text](./visualize/public_test_img_7.jpg) |

**Evaluation metric:** The evaluation metric is CER, which represents the percentage of characters in the label file text that are incorrectly predicted. The lower the CER, the more accurate the recognition model.

$$ CER = \frac{S+D+I}{N} $$

- S (substitutions): Number of characters that need to be replaced to transform the prediction into the ground truth.
- D (deletions): Number of characters that need to be deleted to transform the prediction into the ground truth.
- I (insertions): Number of characters that need to be inserted to transform the prediction into the ground truth.
- N: Number of characters in the word.

## Data Preprocessing

Image convert to **grayscale color**, resized image to `(WIDTH, HEIGHT) = (100,32)`.

## Model

The model as built is a hybrid of Shi et al.'s CRNN architecture ([arXiv:1507.0571](https://arxiv.org/abs/1507.05717)) and the VGG deep convnet, which reduces the number of parameters by stacking pairs of small 3x3 kernels. In addition, the pooling is also limited in the horizontal direction to preserve resolution for character recognition. There must be at least one horizontal element per character.

## Visualize result

## Running on docker

### 1. Training model

```
docker build -t ocr_train .

docker run --name ocr_train ocr_train
```

### 2. Predict

Open terminal on docker

```
python './src/predict.py'
```

The result are saved to file `'/app/'prediction.txt`

## Future work

- Implement paper Robust Scene Text Recognition with Automatic Rectification ([RARE](https://arxiv.org/abs/1603.03915)) to address irregular text. RARE is a specially designed deep neural network, which consists of a Spatial Transformer Network (STN) and a Sequence Recognition Network (SRN).

- Try SOTA model like **Vision Transformer** (ViT), **Swin Transformer**,...  
