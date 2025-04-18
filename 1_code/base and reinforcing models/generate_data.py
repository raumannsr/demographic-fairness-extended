import numpy as np
from keras.utils import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator


def isic_generate_data_stl(directory, augmentation, batch_size, file_list, label_1):
    i = 0
    while True:
        image_batch = []
        label_1_batch = []
        for b in range(batch_size):
            if i == (len(file_list)):
                i = 0
            img = load_img(directory + '/' + file_list.iloc[i] + '.JPG', grayscale=False)
            img = img_to_array(img)

            if augmentation:
                datagen = ImageDataGenerator(
                    rotation_range=360,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.2,
                    zoom_range=0.2,
                    channel_shift_range=20,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode="nearest")
                img = datagen.random_transform(img)
                img = img / 255.0
            else:
                img = img / 255.0

            image_batch.append(img)
            label_1_batch.append(label_1.iloc[i])
            i = i + 1

        yield np.asarray(image_batch), np.asarray(label_1_batch)


def isic_generate_data_mtl(directory, augmentation, batch_size, file_list, label_1, label_2):
    i = 0
    while True:
        image_batch = []
        label_1_batch = []
        label_2_batch = []
        for b in range(batch_size):
            if i == (len(file_list)):
                i = 0
            img = load_img(directory + '/' + file_list.iloc[i] + '.JPG', grayscale=False)
            img = img_to_array(img)
            if augmentation:
                datagen = ImageDataGenerator(
                    rotation_range=360,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=0.2,
                    zoom_range=0.2,
                    channel_shift_range=20,
                    horizontal_flip=True,
                    vertical_flip=True,
                    fill_mode="nearest")
                img = datagen.random_transform(img)
                img = img / 255.0
            else:
                img = img / 255.0

            image_batch.append(img)
            label_1_batch.append(label_1.iloc[i])
            label_2_batch.append(label_2[i])
            i = i + 1

        yield (
            np.asarray(image_batch),
            {'out_class': np.asarray(label_1_batch), 'out_demographic': np.asarray(label_2_batch)})
