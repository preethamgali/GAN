import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt

z_latent_size = 100
with tf.device('/GPU:0'):
    initializer = tf.contrib.layers.xavier_initializer()
    weights_bias = {
      'g_hl1_weights' : tf.Variable(initializer(shape = (z_latent_size, 128) ,dtype=tf.float64)),
      'g_hl1_bias' : tf.Variable(initializer(shape = (1, 128) ,dtype=tf.float64)),

      'g_hl2_weights' : tf.Variable(initializer(shape = (128, 28*28) ,dtype=tf.float64)),
      'g_hl2_bias' : tf.Variable(initializer(shape = (1, 28*28) ,dtype=tf.float64)),

      'd_hl1_weights' : tf.Variable(initializer(shape = (28*28, 128) ,dtype=tf.float64)),
      'd_hl1_bias' : tf.Variable(initializer(shape = (1, 128) ,dtype=tf.float64)),

      'd_out_weights' : tf.Variable(initializer(shape = (128, 1) ,dtype=tf.float64)),
      'd_out_bias' : tf.Variable(initializer(shape = (1, 1) ,dtype=tf.float64)),
    }

def scale_data(data):
    data = data/255
    return data*2 - 1

def gan_input():
    z = tf.placeholder(dtype= tf.float64, shape= (None,100))
    img = tf.placeholder(dtype= tf.float64, shape= (None,28*28))
    return z,img

def generator(z,weights_bias, reuse):
    with tf.variable_scope('gen',reuse=reuse):
        g_hl1 = tf.matmul(z ,weights_bias['g_hl1_weights']) + weights_bias['g_hl1_bias']
        g_hl1_activation = tf.nn.leaky_relu(g_hl1)
        g_hl2 = tf.matmul(g_hl1_activation ,weights_bias['g_hl2_weights']) + weights_bias['g_hl2_bias']
        g_out = tf.nn.tanh(g_hl2)
    return g_out

def discriminator(img,weights_bias, reuse):
    with tf.variable_scope('dis', reuse= reuse):
      d_hl1 = tf.matmul(img ,weights_bias['d_hl1_weights']) + weights_bias['d_hl1_bias']
      d_hl1_activation = tf.nn.leaky_relu(d_hl1)
      logits = tf.matmul(d_hl1_activation , weights_bias['d_out_weights']) + weights_bias['d_out_bias']
    return logits
    
z,img = gan_input()
gen = generator(z,weights_bias, reuse= False)
generate_img = generator(z,weights_bias, reuse= True)
d_fake = discriminator(gen,weights_bias, reuse=False)
d_real = discriminator(img,weights_bias, reuse= True)

d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels= tf.zeros_like(d_fake), logits=d_fake))
d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels= tf.ones_like(d_real), logits=d_real))
d_loss = d_fake_loss + d_real_loss

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels= tf.ones_like(d_fake), logits= d_fake))

g_vars = [weights_bias['g_hl1_weights'], weights_bias['g_hl1_bias'],weights_bias['g_hl2_weights'], weights_bias['g_hl2_bias']]
d_vars = [weights_bias['d_hl1_weights'], weights_bias['d_hl1_bias'],weights_bias['d_out_weights'], weights_bias['d_out_bias']]

train_d = tf.train.AdamOptimizer().minimize(d_loss, var_list= d_vars)
train_g = tf.train.AdamOptimizer().minimize(g_loss, var_list= g_vars)

batch_size = 100
epochs = 100

(train_data,test_data) = mnist.load_data()
data_set = scale_data(np.asarray([data.reshape(28*28) for data in train_data[0]]))

with tf.device('/GPU:0'):
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  batches = [data_set[i:i+batch_size] for i in range(0,data_set.shape[0],batch_size)]
  for e in range(epochs):
    for batch in batches:
      z_latent = np.random.uniform(-1,1,size=(batch_size, 100))
      sess.run(train_d , feed_dict= {z:z_latent, img:batch})
      sess.run(train_g, feed_dict= {z:z_latent})

    if e%(epochs/10) == 0:
      d_error = sess.run(d_loss,feed_dict= {z:z_latent, img:batch})
      g_error = sess.run(g_loss,feed_dict= {z:z_latent})
      print('e:',e,
          'd_error:',np.mean(d_error),
          'g_error:',np.mean(g_error))

z_latent = np.random.uniform(-1,1,size=(1, 100))
genrated_img = sess.run(generate_img, feed_dict= {z:z_latent}).reshape((28,28))
plt.imshow(genrated_img)
