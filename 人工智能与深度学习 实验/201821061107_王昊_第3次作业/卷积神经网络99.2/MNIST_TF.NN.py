import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 导入 MNIST 数据集
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])  # 输入的数据占位符:存放待识别的图片
y_actual = tf.placeholder(tf.float32, shape=[None, 10])  # 输入的标签占位符:放0~9 这10个数字

# 定义一个函数，用于初始化所有的权值 W
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) # 从截断的正态分布中输出随机值
    return tf.Variable(initial)


# 定义一个函数，用于初始化所有的偏置项 b
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape) # 创建一个常量tensor
    return tf.Variable(initial)


# 定义一个函数，用于构建[卷积层]
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  #x是图片的所有参数，W是此卷积层的权重，然后定义步长strides=[1,1,1,1]值，strides[0]和strides[3]的两个1是默认值，中间两个1代表padding时在x方向运动一步，y方向运动一步，padding采用的方式是SAME。


# 定义一个函数，用于构建[池化层]
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')# 为了得到更多的图片信息，padding时我们选的是一次一步，也就是strides[1]=strides[2]=1，这样得到的图片尺寸没有变化，而我们希望压缩一下图片也就是参数能少一些从而减小系统的复杂度，因此我们采用pooling来稀疏化参数，也就是卷积神经网络中所谓的下采样层。pooling 有两种，一种是最大值池化，一种是平均值池化，本例采用的是最大值池化tf.max_pool()。池化的核函数大小为2x2，因此ksize=[1,2,2,1]，步长为2，因此strides=[1,2,2,1]:


# 构建网络
x_image = tf.reshape(x, [-1, 28, 28, 1])  # 转换输入数据shape,以便于用于网络中
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 第一个卷积层
h_pool1 = max_pool(h_conv1)  # 第一个池化层

W_conv2 = weight_variable([5, 5, 32, 64]) # 接着呢，同样的形式我们定义第二层卷积，本层我们的输入就是上一层的输出，本层我们的卷积核patch的大小是5x5，有32个featuremap所以输入就是32，输出呢我们定为64
b_conv2 = bias_variable([64]) 
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 第二个卷积层
h_pool2 = max_pool(h_conv2)  # 第二个池化层:输出大小为7x7x64


# 建立全连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024]) #weight_variable的shape输入就是第二个卷积层展平了的输出大小: 7x7x64，后面的输出size我们继续扩大，定为1024
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # 通过tf.reshape()将h_pool2的输出值从一个三维的变为一维的数据, -1表示先不考虑输入图片例子维度, 将上一个输出结果展平.
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # 将展平后的h_pool2_flat与本层的W_fc1相乘（注意这个时候不是卷积了）

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # dropout层

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_predict = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # softmax层

cross_entropy = -tf.reduce_sum(y_actual * tf.log(y_predict))  # 利用交叉熵损失函数来定义我们的cost function
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)  # 梯度下降法
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_actual, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 精确度计算
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:  # 训练100次，验证一次
        train_acc = accuracy.eval(feed_dict={x: batch[0], y_actual: batch[1], keep_prob: 1.0})
        print('step', i, 'training accuracy', train_acc)
        train_step.run(feed_dict={x: batch[0], y_actual: batch[1], keep_prob: 0.5})


#由于我自己的电脑显存不够,所以把数据集分10次喂给训练器,最后取平均的正确率
test_acc = 0
for i in range(10):
    test_acc += accuracy.eval(feed_dict={x: mnist.test.images[1000*i:1000*(i+1)], y_actual: mnist.test.labels[1000*i:1000*(i+1)], keep_prob: 1.0})
test_acc /= 10
#test_acc = accuracy.eval(feed_dict={x: mnist.test.images, y_actual: mnist.test.labels, keep_prob: 1.0})

print("测试准确度是:", test_acc)