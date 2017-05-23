"""
Linear Regression on diabetes dataset.
"""

import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
# from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split


# Params
num_epochs = 50000
num_features  = 10

def main(_):
    # Load diabetes dataset
    diabetes = load_diabetes()
    diabetes_data = diabetes.data[:,:num_features]
    # diabetes_data = normalize(diabetes_data)
    diabetes_target = diabetes.target.reshape([diabetes.target.shape[0], 1])
    x_train, x_test, y_train, y_test = train_test_split(
        diabetes_data,
        diabetes_target,
        train_size=0.8,
        random_state=1
    )

    x = tf.placeholder(tf.float32, shape=[None, num_features])
    y = tf.placeholder(tf.float32, shape=[None, 1])

    W = tf.Variable(tf.random_normal([num_features, 1]))
    b = tf.Variable(tf.random_normal([1]))

    y_prediction = tf.matmul(x, W) + b

    loss = tf.reduce_mean(tf.square(y - y_prediction))
    optimizer = tf.train.GradientDescentOptimizer(.05).minimize(loss)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for i in range(num_epochs):
        sess.run(optimizer, feed_dict={x: x_train, y: y_train})
        if ((i+1)%5000==0 or (i+1)<10):
            print("Iteration %s: loss = %s" % (i+1, loss.eval(feed_dict={
                x: x_train,
                y: y_train
            })))

    print("Weight: \n%s"% sess.run(W).T)
    print("Bias: \n%s"% sess.run(b).T)

    prediction_differences = np.absolute(
        sess.run(y_prediction, feed_dict={x: x_test}) - y_test
    )
    print("Diabetes value are off by average %s." % np.mean(prediction_differences))

    prediction_differences = sess.run(y_prediction, feed_dict={x: x_test}) - y_test
    sns.distplot(prediction_differences)
    plt.title("Prediction Difference Distribution")
    plt.show()

if __name__ == '__main__':
    tf.app.run(main=main)
