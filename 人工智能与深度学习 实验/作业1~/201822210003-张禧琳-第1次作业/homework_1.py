#import modules
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist=input_data.read_data_sets("./data", one_hot=True)
#networks parameters
### START CODE HERE ###
W=tf.Variable(tf.zeros([784, 10]))
b=tf.Variable(tf.zeros([10]))
### END CODE HERE ###

#define placeholder for inputs to network
### START CODE HERE ###
x=tf.placeholder(tf.float32, shape=[None, 784])
y_=tf.placeholder(tf.float32, shape=[None, 10])
### END CODE HERE ###

#add layer y=softmax(w*x+b)
### START CODE HERE ###
y=tf.nn.softmax(tf.matmul(x, W) + b)
### END CODE HERE ###

#Define loss and optimizer, minimize the squared error

### START CODE HERE ###
#define cross entropy as loss
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
#define optimizer
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
### END CODE HERE ###


with tf.Session() as sess:
    writer = tf.summary.FileWriter("logs/", sess.graph)
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        batch_xs,batch_ys=mnist.train.next_batch(100)

	##run optimizer
	### START CODE HERE ###
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys})
	### END CODE HERE ###

        if i%20==0:
            correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
            accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
            print(accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels}))
