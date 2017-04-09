from keras.models import load_model

model = load_model('modelGpu')
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

from matplotlib import pyplot as plt

sample_index = 10

plt.imshow(X_train[sample_index])
print y_train[sample_index]
plt.savefig('image.png')

X_train = X_train[sample_index].reshape(1, 28, 28, 1)

print model.predict_on_batch(X_train)
