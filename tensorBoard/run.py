import tensorflow as tf 
import numpy
import pandas as pd
df=pd.read_csv('/dmhack2017/tensorBoard/tensorflow_test_input.tsv', sep='\t', header=None)
d = df.values
l = pd.read_csv('/dmhack2017/tensorBoard/tensorflow_test_label.tsv', sep='\t', header=None)
labels = l.values
data = numpy.float32(d)
labels = numpy.array(l,'str')
#print data, labels

print(data)

log_dir = '/dmhack2017/tensorBoard/logs'
max_steps = 10000

#tensorflow
sess = tf.InteractiveSession()

embedding = tf.Variable(tf.stack(data), trainable=False, name="embedding")

tf.global_variables_initializer().run()

saver = tf.train.Saver()
writer = tf.summary.FileWriter(log_dir + '/projector', sess.graph)

saver.save(sess, log_dir+'/model.ckpt', global_step=max_steps)


