from keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from preprocess import load_data

categories = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
(images, labels) = load_data()
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1)


def evaluate_model():
    model = load_model('model.h5')
    model.evaluate(x_test, y_test, verbose=1)


def visualize_prediction():
    model = load_model('model.h5')
    prediction = model.predict(x_test)
    plt.figure(figsize=(9, 9))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(x_test[i])
        plt.xlabel('Actual: ' + categories[y_test[i]] + '\n' + 'Predicted: ' + categories[np.argmax(prediction[i])])
        plt.xticks([])
        plt.yticks([])
    plt.show()


if __name__ == '__main__':
    evaluate_model()
    visualize_prediction()
