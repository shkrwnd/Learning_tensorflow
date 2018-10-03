import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST-data/", one_hot=True)


time_steps = 28
batch_size = 100
input_size = 28
num_layers = 1
num_hidden = 28
num_classes = 10
learning_rate = 0.01

x = tf.placeholder(tf.float32,[None,time_steps,input_size])
y = tf.placeholder(tf.float32,[None,num_classes])

softmax_w = tf.Variable(tf.random_normal([num_hidden,num_classes]))
softmax_b = tf.Variable(tf.random_normal([num_classes]))

def lstm_cell():
    cell = tf.contrib.rnn.BasicLSTMCell(input_size)
    #cell = tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob = 0.1)
    return cell

def rnn(x,softmax_w,softmax_b):
    #x = tf.unstack(x,time_steps,1)
    stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell() for i in range(num_layers)])
    state= stacked_lstm.zero_state(batch_size,tf.float32)
    outputs ,state = tf.nn.dynamic_rnn(cell=stacked_lstm,inputs=x,dtype=tf.float32)

    return tf.matmul(outputs[:,-1,:] ,softmax_w) + softmax_b




logits = rnn(x,softmax_w,softmax_b)

prediction = tf.nn.softmax(logits)
correct_pred = tf.equal(tf.arg_max(prediction,1),tf.arg_max(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        x_ , y_  = mnist.train.next_batch(batch_size)
        x_ =  np.reshape(x_,[batch_size,time_steps,input_size])
        sess.run([train],feed_dict = {x : x_,y: y_})
    
        if i %200 == 0:
            print (i, sess.run([accuracy, loss],feed_dict = {x : x_,y: y_}))
    
    test_len = 100
    test_data = mnist.test.images[:test_len].reshape((-1, time_steps, input_size))
    test_label = mnist.test.labels[:test_len]
    print "****************"
    print sess.run([accuracy,loss], feed_dict={x: test_data, y: test_label})
