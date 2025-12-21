import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import sys
import tensorflow as tf
from tensorflow.keras import layers
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import time

# --- CONFIGURATION ---
BATCH_SIZE = 256
NOISE_DIM = 100
EPOCHS = 50

# --- LOAD DATA (MNIST) ---
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')

train_images = (train_images - 127.5) / 127.5  # Normalize to [-1, 1]
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(60000).batch(BATCH_SIZE)

# --- BUILD GENERATOR ---
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(NOISE_DIM,)))
    model.add(layers.Dense(7 * 7 * 256, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size
    
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

# --- BUILD DISCRIMINATOR ---
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(28, 28, 1)))
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model    

# --- TEST GENERATOR AND DISCRIMINATOR ---

generator = build_generator()
noise = tf.random.normal([1, NOISE_DIM])
generated_image = generator(noise, training=False)
print("Generated image shape:", generated_image.shape)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')
# plt.show()

discriminator = build_discriminator()
decision = discriminator(generated_image)
print("Decision", decision)

# loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(realOutput, fakeOutput):
    real_loss = cross_entropy(tf.ones_like(realOutput), realOutput)
    fake_loss = cross_entropy(tf.zeros_like(fakeOutput), fakeOutput)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

os.makedirs('./GAN/training_checkpoints', exist_ok=True)
checkpoint_dir = './GAN/training_checkpoints'

epoch_var = tf.Variable(0, dtype=tf.int64)
checkpoint = tf.train.Checkpoint(
    generator=generator,
    discriminator=discriminator,
    epoch=epoch_var
)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

# Restore from the latest checkpoint if one exists
if manager.latest_checkpoint:
    checkpoint.restore(manager.latest_checkpoint)
    print(f"Checkpoint restored from {manager.latest_checkpoint}")
else:
    print("Initializing from scratch.")

# We will reuse this seed overtime to visualize progress
NUM_EXAMPLES_TO_GENERATE = 16
seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])

# --- TRAINING ---
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


#--- TRAINING LOOP ---
def train(dataset, epochs):
    # Generate and save images before training starts
    if epoch_var.numpy() == 0:
        generate_and_save_images(generator, 0, seed)

    for epoch in range(epoch_var.numpy(), epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch)

        # Produce images for the GIF as we go
        # display.clear_output(wait=True) # This is for Jupyter notebooks, can be commented out
        generate_and_save_images(generator, epoch + 1, seed)
        epoch_var.assign_add(1)
        if (epoch + 1) % 1 == 0:
            manager.save()

        print(f'Time for epoch {epoch + 1} is {time.time() - start} sec')

def generate_and_save_images(model, epoch, test_input):
    # Create image directory if it doesn't exist
    os.makedirs('./GAN/images', exist_ok=True)

    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('./GAN/images/image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()
    plt.close(fig)

# --- START TRAINING ---
train(train_dataset, EPOCHS)

# Save the final generator model for inference
generator.save('./GAN/generator_final.keras')
print("Generator model saved to ./GAN/generator_final.keras")