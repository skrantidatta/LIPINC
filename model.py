

# import the modules
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
import math
import tensorflow as tf
import dlib
# from google.colab.patches import cv2_imshow
import tensorflow
from tensorflow import math, matmul, reshape, shape, transpose, cast, float32
from tensorflow.keras.layers import Dense, Layer
from keras.backend import softmax
from skimage.metrics import structural_similarity
from  tensorflow.keras import losses
import cv2
from keras import backend as K
from keras.layers import Layer,InputSpec
import keras.layers as kl
from glob import glob
import shutil
from sklearn.metrics import roc_curve, auc
from keras.preprocessing import image
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from skimage.util import random_noise
from sklearn.metrics import roc_auc_score
from sklearn.metrics import  precision_score, recall_score, accuracy_score,average_precision_score
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,Callback
from  matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import concatenate,Dense, Conv2D,Conv3D, MaxPooling2D, MaxPooling3D,Flatten,Input,Activation,add,AveragePooling2D,BatchNormalization,Dropout,Reshape
from sklearn.metrics import  precision_score, recall_score, accuracy_score,classification_report ,confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import AUC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from scipy.special import expit

def LIPINC_model():
  #Input layer
  FrameInput=Input(shape=(8,64, 144,3),name = 'FrameInput')
  ResidueInput = Input(shape=(7,64, 144,3),name = 'ResidueInput')

  # 3D CNN + linear
  def CNN(Input,given_name):
    conv3= (Conv3D(8, kernel_size=(3, 3, 3), activation='relu',padding ="same", kernel_initializer='he_uniform',name= given_name)(Input))
    conv3= (MaxPooling3D(pool_size=(2, 2, 2))(conv3))
    conv3= (BatchNormalization(center=True, scale=True)(conv3))
    conv3= (Dropout(0.5)(conv3))

    conv3= (Flatten()(conv3))
    conv3= (Dense(8*8*3, activation='relu', kernel_initializer='he_uniform')(conv3))
    conv3= (Reshape((8,8,3))(conv3))
    return conv3

  CNNframe = CNN(FrameInput,'Frame')
  CNNresidue = CNN(ResidueInput,'Res')

  # MSTIE

  # Implementing the Scaled-Dot Product Attention
  class DotProductAttention(Layer):
    def __init__(self, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)

    def call(self, queries, keys, values, d_k, mask=None):
        # Scoring the queries against the keys after transposing the latter, and scaling
        scores = matmul(queries, keys, transpose_b=True) / math.sqrt(cast(d_k, float32))

        # Apply mask to the attention scores
        if mask is not None:
            scores += -1e9 * mask

        # Computing the weights by a softmax operation
        weights = softmax(scores)

        # Computing the attention by a weighted sum of the value vectors
        return matmul(weights, values)

  attention1 = DotProductAttention()
  attention2 = DotProductAttention()
  attention3 = DotProductAttention()

  #Color branch
  keys = CNNframe
  values = CNNframe
  queries = CNNresidue

  d_k = queries.shape[1]*queries.shape[2]*queries.shape[3]
  conv_color = attention1(queries, keys, values, d_k)

  #Structure Branch
  keys = CNNresidue
  values = CNNresidue
  queries = CNNframe

  d_k = queries.shape[1]*queries.shape[2]*queries.shape[3]
  conv_res = attention2(queries, keys, values, d_k)

  #Fusion
  keys = conv_color
  values = conv_color
  queries = conv_res

  d_k = queries.shape[1]*queries.shape[2]*queries.shape[3]

  conv_fusion = attention3(queries, keys, values, d_k)

  #Concat
  conv = concatenate([CNNresidue,conv_fusion])

  #MLP
  conv=(Conv2D(filters=64,kernel_size=(3,3), activation="relu",padding="same",kernel_initializer='he_normal')(conv))
  conv=(BatchNormalization()(conv))
  conv=(Conv2D(filters=64,kernel_size=(1,1), activation="relu",padding="same",kernel_initializer='he_normal')(conv))
  conv=(BatchNormalization()(conv))

  conv=(MaxPooling2D(strides=(2, 2),padding="same")(conv))

  conv=(Conv2D(filters=128,kernel_size=(3,3), activation="relu",padding="same",kernel_initializer='he_normal')(conv))
  conv=(BatchNormalization()(conv))
  conv=(Conv2D(filters=128,kernel_size=(1,1), activation="relu",padding="same",kernel_initializer='he_normal')(conv))
  conv=(BatchNormalization()(conv))

  conv=(MaxPooling2D(pool_size=(4, 4),padding="same")(conv))

  conv=(Flatten()(conv))
  conv=(Dense(4096,activation="relu")(conv))
  conv=(Dense(4096,activation="relu")(conv))
  out=(Dense(2, activation="softmax")(conv))

  model = Model(inputs=[FrameInput,ResidueInput], outputs=out)

  #Loss model for inconsistency loss
  loss_model = Model(FrameInput,outputs=model.get_layer('Frame').output)
  # loss_model.summary()

  def similarity(a,b):
    (score, diff) = structural_similarity(a, b, full=True,multichannel =True)
    return score

  def total_loss(FrameInput,loss_model):
    def loss(y_true, y_pred):

      # CCE = K.categorical_crossentropy(y_true, y_pred)

      z = loss_model(FrameInput)
      tot = 0
      for i in range(z.shape[0]):
        a = z[i]
        for j in range(z.shape[0]):
          b = z[j]
          sim = similarity(a,b)
          tot+=sim

      avg_sim = tot/(z.shape[0]*z.shape[0])
      BCE = tf.keras.losses.BinaryCrossentropy(from_logits=False)
      con_loss = BCE(y_true,avg_sim)

      return con_loss

    return loss
  custom_loss = total_loss(FrameInput,loss_model)

  opt1=tf.keras.optimizers.Adam(learning_rate=0.001,epsilon=0.1)
  model.compile(optimizer=opt1,loss = ["categorical_crossentropy",total_loss],loss_weights=[1,5])



  return model


