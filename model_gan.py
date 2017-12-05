import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

if not os.path.exists('out/'):
	os.makedirs('out/')

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', 'MNIST-data/', 'Directory for storing data')
flags.DEFINE_integer('steps','10000','noof stpes')
mb_size = 32
class Model():
	def __init__(self):
		self.g_w1  = tf.Variable(tf.truncated_normal([64,128],stddev = 0.1))
		self.g_b1 = tf.Variable(tf.constant(0.1,shape = [128]))
		self.g_w2  = tf.Variable(tf.truncated_normal([128,784],stddev = 0.1))
		self.g_b2 = tf.Variable(tf.constant(0.1,shape=[784]))

		self.d_w1  = tf.Variable(tf.truncated_normal([784,128],stddev = 0.1))
		self.d_b1 = tf.Variable(tf.constant(0.1,shape = [128]))
		self.d_w2  = tf.Variable(tf.truncated_normal([128,1],stddev = 0.1))
		self.d_b2 = tf.Variable(tf.constant(0.1,shape=[1]))
		self.g_list = [self.g_w1,self.g_w2,self.g_b1,self.g_b2]
		self.d_list = [self.d_w1,self.d_w2,self.d_b1,self.d_b2]
		


	def init_model_var(self):
		self.init = tf.global_variables_initializer()
		return self.init

	def generator(self,x):
		self.g_fc1 = tf.nn.relu(tf.matmul(x,self.g_w1) + self.g_b1)
		self.g_fc2 = tf.nn.sigmoid(tf.matmul(self.g_fc1,self.g_w2) + self.g_b2)
		return self.g_fc2

	def discriminator(self,x):
		self.d_fc1 = tf.nn.relu(tf.matmul(x,self.d_w1) + self.d_b1)
		self.d_fc2 = tf.matmul(self.d_fc1,self.d_w2) + self.d_b2
		return self.d_fc2

	def loss(self,y,y_):
		#   y_    :   label placeholder

		pass

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

mnist = input_data.read_data_sets(FLAGS.data_dir,one_hot=True)
x = tf.placeholder(tf.float32,[None,784])
z = tf.placeholder(tf.float32,[None,64])

model = Model()
g_sample = model.generator(z)
d_real = model.discriminator(x)
d_fake = model.discriminator(g_sample)


d_target = 1./mb_size
g_target = 1./(mb_size*2)

Z = tf.reduce_sum(tf.exp(-d_real)) + tf.reduce_sum(tf.exp(-d_fake))


g_loss = tf.reduce_sum(g_target * d_real) + tf.reduce_sum(g_target * d_fake) + tf.log(Z + 1e-8)
d_loss = tf.reduce_sum(d_target * d_real) + tf.log(Z + 1e-8)


g_train = tf.train.AdamOptimizer(1e-3).minimize(g_loss,var_list=model.g_list)
d_train = tf.train.AdamOptimizer(1e-3).minimize(d_loss,var_list=model.d_list)


it = 0
sess = tf.Session()
sess.run(model.init_model_var())

for i in range(10000):
	
	batch_xs ,batch_ys = mnist.train.next_batch(mb_size)
	z_sample = sample_z(mb_size,64)

	_ , g_loss_c = sess.run([g_train,g_loss],feed_dict={x:batch_xs,z:z_sample})
	_ , d_loss_c = sess.run([d_train,d_loss],feed_dict={x:batch_xs,z:z_sample})

	if(i%1000 == 0):
		print ("g_loss ={:.4}, d_loss = {:.4} ,  iter = {}".format(g_loss_c,d_loss_c ,i) )
		samples = sess.run(g_sample,feed_dict={z:sample_z(16,64)})
		fig = plot(samples)
		plt.savefig('out/{}.png'.format(str(it).zfill(3)), bbox_inches='tight')
        it += 1
        plt.close(fig)