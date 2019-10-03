import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
from keras.utils import np_utils
import numpy as np
import os
from tqdm import tqdm

import config
import toolkit_file

import image_process

from pprint import pprint

dataset_dir_1 = config.DATASET_DIR_1
dataset_dir_2 = config.DATASET_DIR_2
data_dump = config.DATA_DMP


def generate_file_list_1(dataset_dir):
    imgFileList = [x for x in toolkit_file.get_file_list(
        dataset_dir) if x.endswith('.jpg')]

    dataset_dict_list = []

    for file in imgFileList:
        pic_id = toolkit_file.get_basename(
            file, withExtension=False)
        # ['Cat', 'Dog']
        if pic_id.lower().startswith('cat'):
            group_id = 0
        elif pic_id.lower().startswith('dog'):
            group_id = 1
        else:
            break
        dataset_dict_list.append(
            {'pic_id': pic_id, 'group_id': group_id, 'image_path': file})
    return dataset_dict_list


def generate_file_list_2(dataset_dir):
    imgFileList = [x for x in toolkit_file.get_file_list(
        dataset_dir) if x.endswith('.jpg')]
    dataset_dict_list = []

    for file in imgFileList:
        pic_id = toolkit_file.get_basename(
            file, withExtension=False)
        # ['Cat', 'Dog']
        if file.split(os.sep)[-2].lower() == 'cat':
            group_id = 1
            dataset_dict_list.append(
                {'pic_id': pic_id, 'group_id': group_id, 'image_path': file})
        elif file.split(os.sep)[-2].lower() == 'dog':
            group_id = 0
            dataset_dict_list.append(
                {'pic_id': pic_id, 'group_id': group_id, 'image_path': file})
        else:
            break
    return dataset_dict_list


def generate_file_list():
    dataset_dict_list = generate_file_list_1(dataset_dir_1)
    # dataset_dict_list += generate_file_list_2(dataset_dir_2)
    cat_count = 0
    dog_count = 0
    for x in dataset_dict_list:
        if x['group_id'] == 0:
            cat_count += 1
        elif x['group_id'] == 1 :
            dog_count += 1
        else:
            raise 
    print('Cat count: {}'.format(cat_count))
    print('Dog count: {}'.format(dog_count))
    return dataset_dict_list


def read_img(image_path_list):
    # x_dataset = np.array([image_process.image_process(x)
    #                       for x in image_path_list])

    x_dataset = []
    for x in tqdm(image_path_list):
        try:
            x_dataset.append(image_process.image_process(x))
        except Exception as e:
            print('Error processing: {}'.format(x))
            print(e)
        else:
            pass
    x_dataset = np.array(x_dataset) / 255

    # x_dataset = np_utils.normalize(x_dataset)
    return x_dataset


def dump_dataset(x_dataset, y_dataset):
    dataset = []
    for x in tqdm(range(len(y_dataset))):
        img_data = x_dataset[x]
        label = y_dataset[x]
        dataset.append((img_data, label))
    dataset = np.array(dataset)
    np.random.shuffle(dataset)
    np.save(data_dump, dataset)


if __name__ == '__main__':
    print('Processing...')
    dataset_dict_list = generate_file_list()
    print('Reading image...')
    x_dataset = read_img([x['image_path'] for x in dataset_dict_list])
    y_dataset = np_utils.to_categorical(
        [x['group_id'] for x in dataset_dict_list])

    print('dumping numpy...')
    print('x_dataset: {}'.format(x_dataset.shape))
    print('y_dataset: {}'.format(y_dataset.shape))
    dump_dataset(x_dataset, y_dataset)
