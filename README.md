# OCR Vienamese Recognization

# Chạy chương trình

## 1. Huấn luyện mô hình

Command
---
```
docker build -t ocr_train . 

docker run --name ocr_train ocr_train
```


## 2. Predict 

Open terminal trên docker

Command
---
```
python './src/predict.py'
```
Kết quả được lưu vào file `'/app/'prediction.txt`