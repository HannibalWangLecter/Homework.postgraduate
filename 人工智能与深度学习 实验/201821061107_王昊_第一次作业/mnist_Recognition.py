import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


print("Starting downloading...")
mnist = input_data.read_data_sets("data", one_hot=True)
print("Finishing downloading...")
model_path = "saver/lenet_model.ckpt"

def create_placeholder(n_H, n_W, n_C, n_y):
    # '''
    # n_H: the hight of image
    # n_W: the width of image
    # n_C: the image's channal numbers
    # n_y: the class we need to classify
    # '''
    with tf.name_scope('Inputs'):
        X = tf.placeholder(tf.float32, shape = [None, n_H, n_W, n_C], name = "X")#data
        Y = tf.placeholder(tf.float32, shape = [None, n_y], name = "Y")#label
        return X, Y

def initialize_parameters():
    with tf.name_scope('W1'):
        W1 = tf.get_variable(name = "W1", dtype = tf.float32, shape = [5, 5, 1, 6],
                        initializer = tf.contrib.layers.xavier_initializer(seed = 0))

    # '''
    # W1_1 average pool ksize = (1, 2, 2, 1) strides = 2
    # '''
    with tf.name_scope('W2'):
        W2 = tf.get_variable(name = "W2", dtype = tf.float32, shape = [5, 5, 6, 16],
                        initializer = tf.contrib.layers.xavier_initializer(seed = 0))

    # '''
    # W2_2 average pool ksize = (1, 2, 2, 1) strides = 2
    # '''
    parameters = {
        "W1": W1,
        "W2": W2
    }

    return parameters

def forward_propagation(X, parameters):
    # '''
    # Implement the forward propagation for the model:
    # (32, 32, 1) -> Conv2D(5*5, strides = 1) -> Avg Pool(2*2, strides = 2) -> Conv2D(5*5, strides = 1) -> Avg Pool(2*2, strides = 2)
    # ->Flatten1(400) -> fullconnected(84) -> softmax(10)
    # '''
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    with tf.name_scope('Conv1'):
        Z1 = tf.nn.conv2d(input = X, filter = W1, strides = (1, 1, 1, 1), padding = "VALID")
        A1 = tf.nn.relu(Z1)
        P1 = tf.nn.avg_pool(value = A1, ksize = (1, 2, 2, 1), strides = (1, 2, 2, 1), padding = "VALID")

    with tf.name_scope('Conv2'):
        Z2 = tf.nn.conv2d(input = P1, filter = W2, strides = (1, 1, 1, 1), padding = "VALID")
        A2 = tf.nn.relu(Z2)
        P2 = tf.nn.avg_pool(value = A2, ksize = (1, 2, 2, 1), strides = (1, 2, 2, 1), padding = "VALID")
    
    with tf.name_scope('Flatten'):#展开超卷积核,reshape
        Z3 = tf.contrib.layers.flatten(P2)
        A3 = tf.nn.relu(Z3)
    
    with tf.name_scope('FullConnected'):#普通连接
        Z4 = tf.contrib.layers.fully_connected(A3, 84)
        A4 = tf.nn.relu(Z4)

    with tf.name_scope('Output'):#LeNet
        Z5 = tf.contrib.layers.fully_connected(A4, 10, activation_fn = None)

    return Z5

def compute_cost(Z5, Y):
    with tf.name_scope('loss'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z5, labels = Y))
        return cost

def model(mnist, learning_rate = 0.001, batch_size = 128, training_iters = 200000,print_cost = True):#学习率=\alpha-梯度下降
    n_H = 28
    n_W = 28
    n_C = 1
    n_y = 10
    X, Y = create_placeholder(n_H, n_W, n_C, n_y)
    costs = []
    
    parameters = initialize_parameters()

    Z5 = forward_propagation(X, parameters)

    cost = compute_cost(Z5, Y)

    with tf.name_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(Z5, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()  #存储所有的variable

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("logs/", sess.graph)
        sess.run(init)
        step = 1

        while step * batch_size < training_iters:
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_x = batch_x.reshape([128, 28, 28, 1])
            sess.run(optimizer, feed_dict = {X: batch_x, Y: batch_y})

            if step % 10 == 0:
                loss, acc = sess.run([cost, accuracy], feed_dict = {X: batch_x, Y: batch_y})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
                costs.append(loss)
            step += 1

        sess.run(parameters)
        print("Optimization finished!")

        save_path = saver.save(sess, model_path) #保存到本地
        print("Model saved in file: %s" % save_path)

        print("Testing Accuracy:", \
                sess.run(accuracy, feed_dict={
                                            X: mnist.test.images[:1024].reshape(1024, 28, 28, 1),
                                            Y: mnist.test.labels[:1024]
                                            }))
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (pre hundreds)')
        plt.title("Learning rate is " + str(learning_rate))
        plt.show()
    
    return parameters

def detector(mnist):
    X = tf.placeholder(tf.float32, shape = [None, 28, 28, 1], name = "X")
    parameters = initialize_parameters()
    Z5 = forward_propagation(X, parameters)
    
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, model_path)
    print("Model restored!")

    pred = tf.argmax(Z5, 1)

    check_id = 1
    loop = True

    while loop:
        print("Input the image number you want to check ->")
        print("### if you input negative number, then the program will be over. ###")
        
        check_id = int(input())
        if check_id < 0:
            break
        
        X_test = mnist.test.images[check_id].reshape(1, 28, 28, 1)
        
        result = sess.run(pred, feed_dict = {X: X_test})

        print("The model predict this number is %d" % (result[0]))

        X_test = X_test.reshape(28, 28)
        plt.imshow(X_test)
        plt.show()

model(mnist)