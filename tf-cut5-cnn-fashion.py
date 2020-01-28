import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
import itertools

fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

K = len(set(y_train))

i = Input(shape=x_train[0].shape)
x = Conv2D(32, (3, 3), strides=2, activation='relu')(i)
x = Conv2D(64, (3, 3), strides=2, activation='relu')(x)
x = Conv2D(128, (3, 3), strides=2, activation='relu')(x)
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(K, activation='softmax')(x)

"""
model = Model(i, x)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
r = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=40)

model.save('cnn-fashion.h5')        # saving trained model
"""

model = tf.keras.models.load_model('cnn-fashion.h5')


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized conf matrix')
    else:
        print('confusion matr witO normalization')

    print(cm)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.xticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center',
                 color='white' if cm[i, j] > threshold else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted')
    plt.show()


# testing trained model
p_test = model.predict(x_test).argmax(axis=1)
cm = confusion_matrix(y_test, p_test)
#plot_confusion_matrix(cm, list(range(10)))


labels = '''T-shirt/top
Trouser
Pullover
Dress
Coat
Sandal
Shirt
Sneaker
Bag
Ankle boot'''.split()


"""
# converting the trained model into tflite model -------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('converted_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
# ------------------------------------------- converting done
"""


# shows 10 pieces of mispredicted tests
# for a in range(10):
# misclassified_idx = np.where(p_test != y_test)[0]
# e = np.random.choice(misclassified_idx)
# plt.imshow(x_test[e].reshape(28, 28), cmap='gray')
# plt.title('True labels: %s , Predicted: %s' % (y_test[e], p_test[e]))
# plt.show()


# taking image from folder by using opencv to have the image in gray color scale
import cv2 as cv
im2 = cv.imread('dress.png')
grayIm = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)

grayIm = tf.cast(grayIm, tf.float32)
arr = np.array(grayIm)
arr = np.expand_dims(arr, axis=0)
arr = np.expand_dims(arr, axis=3)
# print(arr.shape)


# misclassified_idx = np.where(p_test != y_test)[0]
# e = np.random.choice(misclassified_idx)
# plt.imshow(x_test[e].reshape(28, 28), cmap='gray')
# # plt.title('True labels: %s , Predicted: %s' % (y_test[e], p_test[e]))
# plt.show()

# predicting taken image which is converted into array
p_test = model.predict(arr)

# plt.imshow(p_test, cmap='gray')
# # plt.title('True labels: %s , Predicted: %s' % (y_test[e], p_test[e]))
# plt.show()

# having classification
misclassified_idx = np.where(p_test != y_test)[0]
e = np.random.choice(misclassified_idx)

# below codes are for naming the classified image
c = 0
for q in p_test[e]:
    if q != 1.:
        c += 1
    else:
        coef = c

outfit_list = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(outfit_list[coef])

plt.imshow(grayIm, cmap='gray')
plt.title('Predicted: ' + outfit_list[coef])
plt.show()


