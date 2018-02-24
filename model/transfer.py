import tensorflow as tf

def transferlearning(out, is_training, data):
   channels = [64,128,256]
   fs = [7,5,3,3,3]
   ss = [2,1,1,1,1]
   cold=1
   weight={}
   for i, c in enumerate(channels):
        with tf.variable_scope('block_{}'.format(i+1)):
            weight["w{0}".format(i)] = tf.get_variable("W{0}".format(i), [fs[i],fs[i],cold,c], initializer = tf.contrib.layers.xavier_initializer(seed=0))
            #out = tf.nn.conv2d(out,data[i*5], strides=[1,ss[i],ss[i],1], padding='SAME')
            out = tf.nn.conv2d(out,weight["w{0}".format(i)], strides=[1,ss[i],ss[i],1], padding='SAME')
            out = tf.contrib.layers.batch_norm(out,decay=0.9, center=True,scale=True,is_training=is_training,scope="cnn-batch-norm")
            tf.nn.max_pool(out, ksize=[1,2,2,1], strides=[1,2,2,1],padding = 'VALID')

            cold=c
   return out
