import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt

def scale_data(data):
    data = data/255
    return data*2 - 1

def gan_input():
    z = tf.placeholder(dtype= tf.float64, shape= (None,100))
    img = tf.placeholder(dtype= tf.float64, shape= (None,28*28))
    return z,img

def generator(z, reuse):
    print(z)
    with tf.variable_scope('gen',reuse=reuse):
      d1 = tf.layers.dense(z,128)
      d1 = tf.maximum(d1,d1*0.2)
      d2 = tf.layers.dense(d1,28*28)
      d2 = tf.tanh(d2)
    return d2

def discriminator(img,reuse):
    with tf.variable_scope('dis', reuse= reuse):
      d1 = tf.layers.dense(img,128)
      d1 = tf.maximum(d1,d1*0.2)
      d2 = tf.layers.dense(d1,1)
      # d2 is logits
    return d2
    
z,img = gan_input()
gen = generator(z,reuse= False)
generate_img = generator(z,reuse= True)
d_fake = discriminator(gen, reuse=False)
d_real = discriminator(img, reuse= True)

d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels= tf.zeros_like(d_fake), logits=d_fake))
d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels= tf.ones_like(d_real), logits=d_real))
d_loss = d_fake_loss + d_real_loss

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels= tf.ones_like(d_fake), logits= d_fake))

t_vars = tf.trainable_variables()
g_vars = [var for var in t_vars if 'gen' in var.name]
d_vars = [var for var in t_vars if 'dis' in var.name]

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
