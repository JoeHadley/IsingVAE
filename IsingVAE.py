import tensorflow as tf
import matplotlib.pyplot as plt

import os
from google.colab import files
import numpy as np

# Change the current working directory to the folder
os.chdir('cropped')
uploaded = files.upload()

# preprocess
image_size = 28

def preprocess(image):
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image = image / 255.0
    #image = tf.reshape(image, shape = (image_size, image_size, 3,))
    return image


# build the model

latent_dim = 512
from keras.models import Sequential, Model

from keras.layers import Dense, Conv2D, Conv2DTranspose, Input, Flatten, BatchNormalization, Lambda, Reshape, Activation
from keras.layers import LeakyReLU
from keras.activations import selu
from keras.layers import Multiply, Add
from keras.optimizers import Adam

from keras import backend as K

K.clear_session()

# Build encoder
encoder_input = Input(shape=(28, 28, 1))

x = Conv2D(28, kernel_size=5, activation=LeakyReLU(0.02), strides=1, padding='same')(encoder_input)
x = BatchNormalization()(x)

filter_size = [64, 128, 256, 512]
for i in filter_size:
    x = Conv2D(i, kernel_size=5, activation=LeakyReLU(0.02), strides=2, padding='same')(x)
    x = BatchNormalization()(x)

x = Flatten()(x)
x = Dense(1024, activation='selu')(x)
encoder_output = BatchNormalization()(x)

# sampling layer
mu = Dense(latent_dim)(encoder_output)
log_var = Dense(latent_dim)(encoder_output)

def sampling(args):
    mu, log_var = args
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return mu + K.exp(0.5 * log_var) * epsilon

z = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,))([mu, log_var])

encoder = Model(encoder_input, outputs=[mu, log_var, z], name='encoder')
encoder.summary()

# Build the decoder
decoder = Sequential()
decoder.add(Dense(1024, activation='selu', input_shape=(latent_dim,)))
decoder.add(BatchNormalization())

decoder.add(Dense(8192, activation='selu'))
decoder.add(Reshape((4, 4, 512)))

# Adjust the transpose convolutional layers for 28x28x1
decoder.add(Conv2DTranspose(256, (5, 5), activation=LeakyReLU(0.02), strides=2, padding='same'))
decoder.add(BatchNormalization())

decoder.add(Conv2DTranspose(128, (5, 5), activation=LeakyReLU(0.02), strides=2, padding='same'))
decoder.add(BatchNormalization())

decoder.add(Conv2DTranspose(64, (5, 5), activation=LeakyReLU(0.02), strides=2, padding='same'))
decoder.add(BatchNormalization())

decoder.add(Conv2DTranspose(32, (5, 5), activation=LeakyReLU(0.02), strides=2, padding='same'))
decoder.add(BatchNormalization())

# Adjust the number of filters in the last layer to match the single channel (greyscale)
decoder.add(Conv2DTranspose(1, (5, 5), activation='sigmoid', strides=1, padding='same'))
decoder.add(BatchNormalization())

decoder.summary()

# make loss function 
# vae loss = reconstruction loss + KL div

def reconstruction_loss(y, y_pred):
    return tf.reduce_mean(tf.square(y - y_pred))

def kl_loss(mu, log_var):
    loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var))
    return loss

def vae_loss(y_true, y_pred, mu, log_var):
    return reconstruction_loss(y_true, y_pred) + (1 / (28*28)) * kl_loss(mu, log_var)


# combine encoder and decoder
mu, log_var, z = encoder(encoder_input)
reconstructed = decoder(z)
model = Model(encoder_input, reconstructed, name ="vae")
loss = kl_loss(mu, log_var)
model.add_loss(loss)
model.summary()


# make a function to save images while learning
def save_images(model, epoch, step, input_):
    prediction = model.predict(input_)
    fig, axes = plt.subplots(5,5, figsize = (14,14))
    idx = 0
    for row in range(5):
        for column in range(5):
            image = prediction[idx] * 255
            image = image.astype("int32")
            axes[row, column].imshow(image)
            axes[row, column].axis("off")
            idx+=1
    output_path = "output/"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    plt.savefig(output_path + "Epoch_{:04d}_step_{:04d}.jpg".format(epoch, step))
    plt.close()

# train vae

from keras.optimizers import Adam

random_vector = tf.random.normal(shape = (25, latent_dim,))
save_images(decoder, 0, 0, random_vector)

mse_losses = []
kl_losses = []

optimizer = Adam(0.0001, 0.5)
epochs = 1

for epoch in range(1, epochs + 1):
    print("Epoch: ", epoch)
    for step, training_batch in enumerate(training_dataset):
        with tf.GradientTape() as tape:
            reconstructed = model(training_batch)
            y_true = tf.reshape(training_batch, shape = [-1])
            print(y_true)
            y_pred = tf.reshape(reconstructed, shape = [-1])
            print(y_pred)
            mse_loss = reconstruction_loss(y_true, y_pred)
            mse_losses.append(mse_loss.numpy())
            
            kl = sum(model.losses)
            kl_losses.append(kl.numpy())
            
            train_loss = 0.01 * kl + mse_loss
            
            grads = tape.gradient(train_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            if step % 10 == 0:
                save_images(decoder, epoch, step, random_vector)
            print("Epoch: %s - Step: %s - MSE loss: %s - KL loss: %s" % (epoch, step, mse_loss.numpy(), kl.numpy()))
