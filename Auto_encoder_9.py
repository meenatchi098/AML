import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

inp = layers.Input(shape=(784,))
encoded = layers.Dense(32, activation='relu')(inp)
decoded = layers.Dense(784, activation='sigmoid')(encoded)
autoencoder = models.Model(inp, decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=5, batch_size=256,
                shuffle=True, validation_data=(x_test, x_test))

decoded_imgs = autoencoder.predict(x_test)

plt.figure(figsize=(15, 3))
for i in range(10):
    plt.subplot(2, 10, i+1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    plt.title('original')
    plt.axis('off')

    plt.subplot(2, 10, i+11)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    plt.title('reconstructed')
    plt.axis('off')

plt.tight_layout()
plt.show()
