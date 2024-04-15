import tensorflow as tf
from keras.layers import Input, Dense, Reshape, Flatten, Conv2DTranspose, Conv2D, LeakyReLU, Dropout
from keras.models import Model
import matplotlib.pyplot as plt


def load_dataset(directory, image_size=(64, 64)):
    return tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        label_mode=None,
        color_mode='rgb',  
        batch_size=32,  
        image_size=image_size  
    ) / 255.0  

def build_generator(latent_dim, output_shape=(64, 64, 3)):
   
    input_layer = Input(shape=(latent_dim,))
    x = Dense(128 * 16 * 16)(input_layer)
    x = LeakyReLU(alpha=0.2)(x)
    x = Reshape((16, 16, 128))(x)
    
  
    x = Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same')(x)
    output_layer = tf.keras.layers.Activation('tanh')(x)
    
    return Model(inputs=input_layer, outputs=output_layer)

def build_discriminator(input_shape=(64, 64, 3)):
    input_layer = Input(shape=input_shape)
    
    x = Conv2D(64, (3, 3), strides=(2, 2), padding='same')(input_layer)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)
    x = Conv2D(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    
    return Model(inputs=input_layer, outputs=x)

def train_model(generator, discriminator, dataset, latent_dim, epochs=100):
    dataset = dataset.shuffle(10000).batch(32)

    discriminator_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    generator_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    

    for epoch in range(epochs):
        for real_images in dataset:
            
            discriminator_loss_value = train_step(discriminator, generator, real_images, discriminator_loss, discriminator_optimizer)
            
            generator_loss_value = train_step(generator, discriminator, real_images, generator_loss, generator_optimizer)
        
        print(f"Epoch {epoch+1}/{epochs}, Discriminator Loss: {discriminator_loss_value}, Generator Loss: {generator_loss_value}")

def train_step(model, other_model, real_images, loss_function, optimizer):
    noise = tf.random.normal([real_images.shape[0], latent_dim])
    with tf.GradientTape() as tape:
        generated_images = model(noise, training=True)
        real_output = other_model(real_images, training=True)
        fake_output = other_model(generated_images, training=True)
        loss = loss_function(real_output, tf.ones_like(real_output)) + loss_function(fake_output, tf.zeros_like(fake_output))
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def generate_images(generator, latent_dim, num_images=10):
    noise = tf.random.normal([num_images, latent_dim])
    generated_images = generator(noise, training=False)
    return generated_images

latent_dim = 100

generator = build_generator(latent_dim)
discriminator = build_discriminator()
generated_images = generate_images(generator, latent_dim=100)


plt.figure(figsize=(10, 10))
for i in range(generated_images.shape[0]):
    plt.subplot(5, 5, i+1)
    plt.imshow(generated_images[i])
    plt.axis('off')
plt.savefig('ProGan.png')
