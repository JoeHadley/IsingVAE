from tensorflow import keras
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from keras.models import Model
from keras.datasets import mnist
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt


(x_train,y_train),(x_test,y_test) = mnist.load_data()


# Normalise
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/255
x_test = x_test/255

# Reshape
img_width = x_train.shape[1]
img_height = x_train.shape[2]
num_channels = 1
x_train = x_train.reshape(x_train.shape[0],img_height,img_width,num_channels)
x_test = x_test.reshape(x_test.shape[0],img_height,img_width,num_channels)
input_shape = (img_height, img_width, num_channels)



# Eager execution hack
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


# Build Model

latent_dim = 2

input_img = Input(shape=input_shape, name='encoder_input')
x = Conv2D(32, 3, padding = 'same', activation = 'relu')(input_img)
x = Conv2D(64, 3, padding = 'same', activation = 'relu',strides = (2, 2))(x)
x = Conv2D(64, 3, padding = 'same', activation = 'relu')(x)
x = Conv2D(64, 3, padding = 'same', activation = 'relu')(x)

conv_shape = K.int_shape(x) # Shape of con to be provided to decoder

# Flatten
x = Flatten()(x)
x = Dense(32, activation = 'relu')(x)

z_mu = Dense(latent_dim, name = 'latent_mu')(x)
z_sigma = Dense(latent_dim, name = 'latent_sigma')(x)




# Reparametrization Trick

def sample_z(args):
  z_mu, z_sigma = args
  eps = K.random_normal(shape=(K.shape(z_mu)[0],K.int_shape(z_mu)[1]))
  return z_mu + K.exp(z_sigma/2)*eps

z = Lambda(sample_z, output_shape=(latent_dim,),name = 'z')([z_mu,z_sigma])

encoder = Model(input_img, [z_mu, z_sigma, z], name = 'encoder')
print(encoder.summary())


decoder_input = Input(shape = (latent_dim, ), name = 'decoder_input')

x = Dense(conv_shape[1]*conv_shape[2]*conv_shape[3], activation = 'relu')(decoder_input)

x = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)


x = Conv2DTranspose(32, 3, padding = 'same', activation = 'relu', strides = (2,2))(x)

x = Conv2DTranspose(num_channels, 3, padding = 'same', activation = 'sigmoid', name = 'decoder_output')(x)

decoder = Model(decoder_input,x,name = 'decoder')
decoder.summary()

z_decoded = decoder(z)


# Custom Loss
class CustomLayer(keras.layers.Layer):

    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        
        # Reconstruction loss (as we used sigmoid activation we can use binarycrossentropy)
        recon_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        
        # KL divergence
        kl_loss = -5e-4 * K.mean(1 + z_sigma - K.square(z_mu) - K.exp(z_sigma), axis=-1)
        #kl_loss = -0.5 * K.prod(K.sum(1 + z_sigma - K.square(z_mu) - K.exp(z_sigma), axis=-1))

        return K.mean(recon_loss + kl_loss)

    # add custom loss to the class
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x


y = CustomLayer()([input_img, z_decoded])



# VAE 

vae = Model(input_img, y, name='vae')

# Compile VAE
vae.compile(optimizer='adam', loss=None)
vae.summary()

# Train autoencoder
vae.fit(x_train, None, epochs = 10, batch_size = 32, validation_split = 0.2)
#


# Visualize results

#Visualize inputs mapped to the Latent space
#Remember that we have encoded inputs to latent space dimension = 2. 
#Extract z_mu --> first parameter in the result of encoder prediction representing mean

mu, _, _ = encoder.predict(x_test)
#Plot dim1 and dim2 for mu
plt.figure(figsize=(10, 10))
plt.scatter(mu[:, 0], mu[:, 1], c=y_test, cmap='brg')
plt.xlabel('dim 1')
plt.ylabel('dim 2')
plt.colorbar()
plt.show()


# Visualize images
#Single decoded image with random input latent vector (of size 1x2)
#Latent space range is about -5 to 5 so pick random values within this range
#Try starting with -1, 1 and slowly go up to -1.5,1.5 and see how it morphs from 
#one image to the other.
sample_vector = np.array([[1,-1]])
decoded_example = decoder.predict(sample_vector)
decoded_example_reshaped = decoded_example.reshape(img_width, img_height)
plt.imshow(decoded_example_reshaped)
