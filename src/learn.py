import numpy as np
import os
import cv2
import tensorflow as tf

class TitanicModel:
    def __init__(self, featureSize=24, onehotSize=18, learningRate=0.01):
        self.__graph = tf.Graph()
        self.__globalStep = tf.Variable(0, trainable=False)
        self.__learningRate = tf.train.exponential_decay(
            learningRate, self.__globalStep, decay_steps=10, decay_rate=0.98, staircase=True)
        self.__input = tf.placeholder(tf.float32, shape=[None, featureSize], name="input")
        self.__label = tf.placeholder(tf.float32, shape=[None], name="label")

        onehotFeatures, otherFeatures = tf.split(self.__input, [onehotSize, featureSize-onehotSize], axis=1)

        onehotFC1 = tf.layers.dense(onehotFeatures, units=onehotSize, activation=tf.nn.relu)
        onehotFC2 = tf.layers.dense(onehotFC1, units=2, activation=tf.nn.relu)
        
        otherFC1 = tf.layers.dense(otherFeatures, units=featureSize-onehotSize, activation=tf.nn.sigmoid)
        otherFC2 = tf.layers.dense(otherFC1, units=featureSize-onehotSize, activation=tf.nn.sigmoid)
        otherFC3 = tf.layers.dense(otherFC2, units=featureSize-onehotSize, activation=tf.nn.sigmoid)
        otherFC4 = tf.layers.dense(otherFC3, units=2, activation=tf.nn.sigmoid)

        concatLayer = tf.concat([onehotFC2, otherFC4], axis=1)
        finalLayer = tf.layers.dense(concatLayer, units=4, activation=tf.nn.sigmoid)
        
        self.__logits = tf.layers.dense(finalLayer, units=2, activation=tf.nn.sigmoid)
        self.__classes = tf.argmax(input=self.__logits, axis=1)
        self.__probabilites = tf.nn.softmax(self.__logits)

        onehotLabels = tf.one_hot(indices=tf.cast(self.__label, tf.int32), depth=2)
        self.__loss = tf.losses.softmax_cross_entropy(onehot_labels=onehotLabels, logits=self.__logits)

        self.__trainOp = tf.contrib.layers.optimize_loss(
            loss=self.__loss,
            global_step=self.__globalStep,
            learning_rate=self.__learningRate,
            optimizer="SGD"
        )

        self.__session = tf.Session()
        self.__session.run(tf.global_variables_initializer())
        self.__saver = tf.train.Saver()

    def fit_on_batch(self, trainData, label):
        feedDict = {
            self.__input: trainData,
            self.__label: label
        }

        _, loss = self.__session.run([self.__trainOp, self.__loss], feed_dict=feedDict)
        return loss

    def predict(self, testData):
        return self.__session.run([self.__classes], feed_dict={self.__input: testData})[0]

    def predict_proba(self, testData):
        classes, proba = self.__session.run([self.__classes, self.__probabilites], feed_dict={self.__input: testData})
        return classes, probabilities

    def save(self, filename):
        self.__saver.save(self.__session, filename)


