import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from keras.models import load_model

import toolkit_file

import data_prep
import config

model_name = config.MODEL_NAME

predict_dir = 'predict'

predictList = toolkit_file.get_file_list(predict_dir)
predict_dataset = data_prep.read_img(predictList)
print(predict_dataset.shape)
predict_dataset = predict_dataset.reshape(predict_dataset.shape[0], predict_dataset.shape[1], predict_dataset.shape[2], -1)

# Predict
model = load_model(model_name)
predict = model.predict(predict_dataset)

print(predict)
print(np.argmax(predict))


import os
import cv2
from matplotlib import pyplot as plt

for no, x in enumerate(predict):
    idx = np.argmax(x)
    print('Confidence: {}%'.format(x[idx]*100))
    guess_img = 'image_{}.jpg'.format(str(idx*80+1).zfill(4))
    guess_img_path = os.path.join(config.DATASET_DIR, guess_img)
    guess_img = cv2.imread(guess_img_path)
    guess_img = cv2.cvtColor(guess_img, cv2.COLOR_BGR2RGB)

    predict_img_path = predictList[no]
    print(predict_img_path)
    predict_img = cv2.imread(predict_img_path)
    predict_img = cv2.cvtColor(predict_img, cv2.COLOR_BGR2RGB)

    tag = ['predict_img' + ' ' + toolkit_file.get_basename(predict_img_path, withExtension=True), 'guess_img']
    for j, i in enumerate([predict_img, guess_img]):
        plt.subplot(1, 2, j+1)
        plt.imshow(i, cmap='gray', vmin = 0, vmax = 255)
        plt.xlabel(tag[j])
    plt.show()
