import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

#Scaling, ReLU, and Softmax
model = keras.Sequential([ keras.layers.Flatten(input_shape=(28,28)), keras.layers.Dense(128, activation=tf.nn.relu), keras.layers.Dense(10, activation=tf.nn.softmax)])

#Optimizer, Loss Function, and Metrics
#Compiling the data
model.compile( optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#Training the Model
model.fit(train_images, train_labels, epochs=5)

#Evaluating the Model with the test data
test_loss, test_acc = model.evaluate(test_images, test_labels)

#Predictions
#This is the probability that the image matches the corresponding label in my set of labels
predictions = model.predict(test_images)
predictions[0]

#To simplify the output
numpy.argmax(predictions[0])

#Verify the prediction by looking at the label

test_labels[0]