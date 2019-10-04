import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os

import numpy as np
from keras.models import load_model

import toolkit_file

import color_card
import data_prep
import config

MODEL_DIR = config.MODEL_DIR

model_name = config.MODEL_NAME
model_file_name = os.path.join(MODEL_DIR, model_name + '.model')

predict_dir = config.PREDICT_DIR
predict_tgt_dir = config.PREDICT_TGT_DIR

predictList = toolkit_file.get_file_list(predict_dir)
predict_dataset = data_prep.read_img(predictList)
print(predict_dataset.shape)
predict_dataset = predict_dataset.reshape(predict_dataset.shape[0], predict_dataset.shape[1], predict_dataset.shape[2], -1)

# Predict
model = load_model(model_file_name)
predict = model.predict(predict_dataset)

predict_dataset = list()

for x in range(len(predictList)):
    predict_dataset.append((predictList[x], predict[x]))
# print(np.argmax(predict))


def predict_show(predict_dataset):
    import cv2
    from matplotlib import pyplot as plt

    fig = plt.figure()

    for no, x in enumerate(predict_dataset[:20]):
        predict_img_path, predictData = x
        figure = fig.add_subplot(5, 4, no + 1)

        idx = np.argmax(predictData)

        if idx == 0:
            str_label = 'Cat'
        else:
            str_label = 'Dog'

        confidence = round(predictData[idx]*100, 2)
        print('Confidence: {}%'.format(confidence))
        
        predict_img = cv2.imread(predict_img_path)
        predict_img = cv2.cvtColor(predict_img, cv2.COLOR_BGR2RGB)

        fileName = toolkit_file.get_basename(predict_img_path, withExtension=True)

        notSureFlag = ''
        notSureFlag = '?' if confidence < 70 else notSureFlag
        notSureFlag = '!' if confidence == 100 else notSureFlag

        tag = '{str_label} {confi}% {fileName}'.format(str_label=str_label+notSureFlag, fileName=fileName, confi=confidence)

        color = color_card.color_card(confidence)

        plt.title(tag,color=color)
        figure.imshow(predict_img, cmap='gray', vmin = 0, vmax = 255)
    plt.show()


def seperate_image(predict_dataset):
    import shutil

    for folder in ['predict_Cat', 'predict_Dog']:
        toolkit_file.create_folder(os.path.join(predict_tgt_dir, folder))

    for no, x in enumerate(predict_dataset):
        predict_img_path, predictData = x

        idx = np.argmax(predictData)

        if idx == 0:
            str_label = 'predict_Cat'
        else:
            str_label = 'predict_Dog'

        confidence = round(predictData[idx]*100, 2)
        fileName = toolkit_file.get_basename(predict_img_path, withExtension=True)

        notSureFlag = ''
        notSureFlag = '?' if confidence < 70 else notSureFlag
        notSureFlag = '!' if confidence == 100 else notSureFlag

        tag = '{str_label} {confi}% {fileName}'.format(str_label=str_label+notSureFlag, fileName=fileName, confi=confidence)
        print(tag)
        shutil.copy(predict_img_path, os.path.join(predict_tgt_dir, str_label))



if __name__ == '__main__':
    # predict_show(predict_dataset)
    seperate_image(predict_dataset)
