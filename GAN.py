import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt

z_latent_size = 100
def normalize_data(data):
    data  = data - np.mean(data)
    data/=np.max(data)
    return data

def de_normalize(data):
    # data = data + 1
    return data

(train_data,test_data) = mnist.load_data()
data_set = normalize_data(np.asarray([data.reshape(1,28*28) for data in train_data[0]]))

with tf.device('/GPU:0'):
    weights_bias = {
      'g_hl1_weights' : tf.Variable(tf.truncated_normal(shape = (z_latent_size, 200) ,dtype=tf.float64)),
      'g_hl1_bias' : tf.Variable(tf.truncated_normal(shape = (1, 200) ,dtype=tf.float64)),

      'g_hl2_weights' : tf.Variable(tf.truncated_normal(shape = (200, 28*28) ,dtype=tf.float64)),
      'g_hl2_bias' : tf.Variable(tf.truncated_normal(shape = (1, 28*28) ,dtype=tf.float64)),

      'd_hl1_weights' : tf.Variable(tf.truncated_normal(shape = (28*28, 200) ,dtype=tf.float64)),
      'd_hl1_bias' : tf.Variable(tf.truncated_normal(shape = (1, 200) ,dtype=tf.float64)),

      'd_hl2_weights' : tf.Variable(tf.truncated_normal(shape = (200, 20) ,dtype=tf.float64)),
      'd_hl2_bias' : tf.Variable(tf.truncated_normal(shape = (1, 20) ,dtype=tf.float64)),

      'd_out_weights' : tf.Variable(tf.truncated_normal(shape = (20, 1) ,dtype=tf.float64)),
      'd_out_bias' : tf.Variable(tf.truncated_normal(shape = (1, 1) ,dtype=tf.float64)),
    }

    def generator(z):
        g_hl1 = tf.matmul(z ,weights_bias['g_hl1_weights']) + weights_bias['g_hl1_bias']
        g_hl1_activation = tf.nn.leaky_relu(g_hl1)
        g_hl2 = tf.matmul(g_hl1_activation ,weights_bias['g_hl2_weights']) + weights_bias['g_hl2_bias']
        g_out = tf.nn.tanh(g_hl2)
        return g_out

    def discriminator(img):
      d_hl1 = tf.matmul(img ,weights_bias['d_hl1_weights']) + weights_bias['d_hl1_bias']
      d_hl1_activation = tf.nn.sigmoid(d_hl1)
      d_hl2 = tf.matmul(d_hl1_activation ,weights_bias['d_hl2_weights']) + weights_bias['d_hl2_bias']
      d_hl2_activation = tf.nn.leaky_relu(d_hl2)
      logits = tf.matmul(d_hl2_activation , weights_bias['d_out_weights']) + weights_bias['d_out_bias']
      return logits

    def loss(z,real_img):
      fake_img = generator(z)
      d_out_real = discriminator(real_img)
      d_out_fake = discriminator(fake_img)

      d_loss_for_real_img =  tf.nn.sigmoid_cross_entropy_with_logits(labels= tf.ones_like(d_out_real), logits= d_out_real)
      d_loss_for_fake_img =  tf.nn.sigmoid_cross_entropy_with_logits(labels= tf.zeros_like(d_out_fake), logits= d_out_fake)
      d_loss = 0.5 * (d_loss_for_real_img + d_loss_for_fake_img)
      g_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels= tf.ones_like(d_out_fake), logits= d_out_fake)
      # # g_loss we need to maximize the value since we only have the method to minimize, we try to minimize the optiste value whihc in turn maximizing

      return d_loss, g_loss

    def train(z, real_img):

      d_loss, g_loss= loss(z, real_img)
      g_vars = [weights_bias['g_hl1_weights'], weights_bias['g_hl1_bias'],weights_bias['g_hl2_weights'], weights_bias['g_hl2_bias']]
      d_vars = [weights_bias['d_hl1_weights'], weights_bias['d_hl1_bias'],weights_bias['d_hl2_weights'], weights_bias['d_hl2_bias'],weights_bias['d_out_weights'], weights_bias['d_out_bias']]
      
      train_discriminator = tf.train.AdamOptimizer().minimize(d_loss, var_list = d_vars)
      train_generator = tf.train.AdamOptimizer().minimize(g_loss, var_list = g_vars)

      return d_loss, g_loss
      
sess =  tf.Session()
n_images = 1
epoch = 100
t_discriminator_loss = 0
t_generator_loss = 0
for e in range(epoch):
    sess.run(tf.global_variables_initializer())
    imgs = data_set[:n_images]
    for i in imgs:
      z_latent = np.random.normal(0,1,(1,z_latent_size))
      discriminator_loss, generator_loss = sess.run(train(z_latent, imgs))
      t_discriminator_loss += discriminator_loss
      t_generator_loss += generator_loss

    if e%(epoch/10) == 0:
      print('epoch:',e)
      print("t_discriminator_loss:",t_discriminator_loss/(n_images * epoch/10))
      print("t_generator_loss:",t_generator_loss/(n_images * epoch/10))
      print()
      t_discriminator_loss = 0
      t_generator_loss = 0

# z_latent = sess.run(tf.truncated_normal((1,z_latent_size)))
generate_img = de_normalize(sess.run(fake_img, feed_dict={ z:z_latent}).reshape(28,28))
plt.imshow(generate_img)
