import tensorflow as tf
import pickle
import numpy
import pandas as pd

data = pickle.load( open ("querycluster.pckl", "rb"))

darray = []

for index in range(len(data)):
    darray.append(data[index][1])

log_dir = '/dmhack2017/tensorBoard/logs'
max_steps = 10000

#tensorflow
sess = tf.InteractiveSession()

embedding = tf.Variable(tf.stack(darray), trainable=False, name="embedding")

tf.global_variables_initializer().run()

saver = tf.train.Saver()
writer = tf.summary.FileWriter(log_dir + '/projector', sess.graph)

saver.save(sess, log_dir+'/model.ckpt', global_step=max_steps)


