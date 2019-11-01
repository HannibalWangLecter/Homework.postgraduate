import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
# 分类函数，根据回归系数和特征向量来计算 Sigmoid的值
def classifyVector(inX, weights):
    '''
    Desc:
        最终的分类函数，根据回归系数和特征向量来计算 Sigmoid 的值，大于0.5函数返回1，否则返回0
    Args:
        inX -- 特征向量，features
        weights -- 根据梯度下降/随机梯度下降 计算得到的回归系数
    Returns:
        如果 prob 计算大于 0.5 函数返回 1
        否则返回 0
    '''
    #c = input('???')
    prob = sigmoid(np.sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

# 打开测试集和训练集,并对数据进行格式化处理
def colicTest():
    '''
    Desc:
        打开测试集和训练集，并对数据进行格式化处理
    Args:
        None
    Returns:
        errorRate -- 分类错误率
    '''
    frTrain = open('data/HorseColicTraining.txt')
    frTest = open('data/HorseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    # 解析训练数据集中的数据特征和Labels
    # trainingSet 中存储训练数据集的特征，trainingLabels 存储训练数据集的样本对应的分类标签
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    # 使用 改进后的 随机梯度下降算法 求得在此数据集上的最佳回归系数 trainWeights
    trainingSet = array(trainingSet)

    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels ,50)


    errorCount = 0
    numTestVec = 0.0
    # 读取 测试数据集 进行测试，计算分类错误的样本条数和最终的错误率

    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights[-1])) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

# 调用 colicTest() 10次并求结果的平均值
def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the error rate is: %f" % (numTests, errorSum/float(numTests)))

if __name__ == '__main__':
    multiTest()