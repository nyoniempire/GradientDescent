import tensorflow as tf
import pandas as pd


dataframe = pd.read_csv("/Users/user/PycharmProjects/theTest/housing.csv",sep=",")

W1 = tf.Variable(4.0,tf.float32)
W2 = tf.Variable(3.0,tf.float32)
b = tf.Variable(5.0,tf.float32)

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

x1_ = dataframe[["housing_median_age"]]
x2_ = dataframe[["total_rooms"]]
y_ = dataframe[["median_house_value"]]

model = W1 * x1 + W2 * x2 + b



cost = tf.reduce_mean(tf.square(model - y))

optimizer = tf.train.GradientDescentOptimizer(0.0000001).minimize(cost)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)
for i in range(10000):

    sess.run(optimizer,{x1:x1_,x2:x2_,y:y_})
    ww1,ww2,bB = sess.run([W1,W2,b])
    print(sess.run([W1,W2,b]))

#they = a * 39.299566694317065 + bB

#print(they)

sess.close()






