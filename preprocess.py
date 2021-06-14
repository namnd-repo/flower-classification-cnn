import os
import numpy as np
import cv2
import pickle

data_path = './flowers'
categories = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']


def store_data():
    data = []

    for category in categories:
        path = os.path.join(data_path, category).replace("\\", "/")
        label = categories.index(category)

        for image_name in os.listdir(path):
            image_path = os.path.join(path, image_name)
            image = cv2.imread(image_path)

            try:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))
                image = np.array(image, dtype=np.float32)

                data.append([image, label])

            except Exception as e:
                pass

    # print(len(data))
    # 3670

    fo = open('data.pickle', 'wb')
    pickle.dump(data, fo)
    fo.close()


def load_data():
    fi = open('data.pickle', 'rb')
    data = pickle.load(fi)
    fi.close()

    np.random.shuffle(data)

    images = []
    labels = []

    for img, label in data:
        images.append(img)
        labels.append(label)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)

    images /= 255.0

    return [images, labels]


if __name__ == "__main__":
    store_data()
    load_data()
