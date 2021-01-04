#!/usr/bin/env python
# coding: utf-8

# # Variables 

# In[1]:

#debug
#pip install 'gast==0.2.2'
#pip install mpl_finance

#data
fix_days = 15
tp       = 0.03
sl       = 0.03
directT  = "B"

parameter_Order = [0,1,2,0,1,3,0,2,1,0,2,3,0,3,1,0,3,2,1,3,0,1]
parameterN      = len(parameter_Order)
trainDays       = ["2000-1-1","2018-1-1"]
valDays         = ["2018-1-1","2020-1-1"]
testDays        = ["2019-6-1","2020-1-1"]

#model
savepath    = "Stockmodel"
subsavepath = "model_8_fixdays_15_tp3_sl3_buy"
epochs = 1000
batchs = 60
recordStep = 1


# # Structure Define
target_label    ='fix_{0}days_tp{1:d}_sl{2:d}_labels_{3}'.format(fix_days, int(tp*100), int(sl*100), directT)
# In[1]:


import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore')

import tensorflow as tf
print("tensorflow version=",tf.__version__)

#pip install tensorflow-addons
#import tensorflow_addons as tfa

# In[2]:


# load GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" #only error


# In[3]:


from tensorflow.contrib import slim as contrib_slim

slim = contrib_slim


# In[32]:
def Instancenorm(x, gamma=1, beta=0):
    # x_shape:[B, H, W, C]
    results = 0.
    eps = 1e-5
    x_mean = tf.reduce_mean(x)
    x_var  = tf.reduce_mean(tf.square(x - x_mean))
    #x_mean = tf.reduce_mean(x, axis=(1, 2), keepdims=True)
    #x_var  = tf.pow(tf.nn.moments(x, axes=(1, 2), keepdims=True), 2)
    x_normalized = (x - x_mean) / tf.sqrt(x_var + eps)
    results = gamma * x_normalized + beta
    return results

def conv2d_nearest(net, num, kernel_size, stride=1, scope="", activation_fn=tf.nn.relu, normalizer_fn=None):
    pad_total = kernel_size - 1 if type(kernel_size)!=list else kernel_size[0] - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    inputs = tf.pad(net, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]], mode="SYMMETRIC")
    return slim.conv2d(inputs, num, kernel_size, stride=stride, padding='VALID', activation_fn=activation_fn, scope=scope)

def avg_pool2d_nearest(net, kernel_size, stride=1, scope=""):
    pad_total = kernel_size - 1 if type(kernel_size)!=list else kernel_size[0] - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    inputs = tf.pad(net, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]], mode="SYMMETRIC")
    return slim.avg_pool2d(inputs, kernel_size, stride=stride, padding='VALID', scope=scope)

def block319_nearest(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None, kernel_size=[3,3], ratio=1):
    
    with tf.compat.v1.variable_scope(scope, 'Block319', [net], reuse=reuse):
        with tf.compat.v1.variable_scope('Branch_0'):
            tower_conv    = conv2d_nearest(net          , 32*ratio, 1, activation_fn=activation_fn, scope='Conv2d_1x1')
        with tf.compat.v1.variable_scope('Branch_1'):
            tower_conv1_0 = conv2d_nearest(net          , 32*ratio, 1, activation_fn=activation_fn, scope='Conv2d_0a_1x1')
            tower_conv1_1 = conv2d_nearest(tower_conv1_0, 32*ratio, kernel_size, activation_fn=activation_fn, scope='Conv2d_0b_{0}x{1}'.format(*kernel_size))
        with tf.compat.v1.variable_scope('Branch_2'):
            tower_conv2_0 = conv2d_nearest(net          , 32*ratio, 1, activation_fn=activation_fn, scope='Conv2d_0a_1x1')
            tower_conv2_1 = conv2d_nearest(tower_conv2_0, 48*ratio, kernel_size, activation_fn=activation_fn, scope='Conv2d_0b_{0}x{1}'.format(*kernel_size))
            tower_conv2_2 = conv2d_nearest(tower_conv2_1, 64*ratio, kernel_size, activation_fn=activation_fn, scope='Conv2d_0c_{0}x{1}'.format(*kernel_size))
        with tf.compat.v1.variable_scope('Branch_3'):
            tower_conv3_0 = conv2d_nearest(net          , 32*ratio, 1, activation_fn=activation_fn, scope='Conv2d_0a_1x1')
            tower_conv3_1 = conv2d_nearest(tower_conv3_0, 48*ratio, kernel_size, activation_fn=activation_fn, scope='Conv2d_0b_{0}x{1}'.format(*kernel_size))
            tower_conv3_2 = conv2d_nearest(tower_conv3_1, 64*ratio, kernel_size, activation_fn=activation_fn, scope='Conv2d_0c_{0}x{1}'.format(*kernel_size))
            tower_conv3_3 = conv2d_nearest(tower_conv3_2, 64*ratio, kernel_size, activation_fn=activation_fn, scope='Conv2d_0d_{0}x{1}'.format(*kernel_size))
        with tf.compat.v1.variable_scope('Branch_4'):
            tower_conv4_0 = conv2d_nearest(net          , 32*ratio, 1, activation_fn=activation_fn, scope='Conv2d_0a_1x1')
            tower_conv4_1 = conv2d_nearest(tower_conv4_0, 48*ratio, kernel_size, activation_fn=activation_fn, scope='Conv2d_0b_{0}x{1}'.format(*kernel_size))
            tower_conv4_2 = conv2d_nearest(tower_conv4_1, 48*ratio, kernel_size, activation_fn=activation_fn, scope='Conv2d_0c_{0}x{1}'.format(*kernel_size))
            tower_conv4_3 = conv2d_nearest(tower_conv4_2, 64*ratio, kernel_size, activation_fn=activation_fn, scope='Conv2d_0d_{0}x{1}'.format(*kernel_size))
            tower_conv4_4 = conv2d_nearest(tower_conv4_3, 64*ratio, kernel_size, activation_fn=activation_fn, scope='Conv2d_0e_{0}x{1}'.format(*kernel_size))
        mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_1, tower_conv2_2, tower_conv3_3, tower_conv4_4])
        up = conv2d_nearest(mixed, net.get_shape()[3], 1, normalizer_fn=None, activation_fn=None, scope='Conv2d_1x1')
        scaled_up = up * scale
        if activation_fn == tf.nn.relu6:
            # Use clip_by_value to simulate bandpass activation.
            scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

        net += scaled_up
        if activation_fn:
            net = activation_fn(net)
        net = Instancenorm(net)
        #net = tfa.layers.InstanceNormalization()(net)
    return net


# In[33]:


def block319(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None, kerenlsize=[3,1]):
    
    with tf.compat.v1.variable_scope(scope, 'Block319', [net], reuse=reuse):
        with tf.compat.v1.variable_scope('Branch_0'):
            tower_conv    = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
        with tf.compat.v1.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, kerenlsize, scope='Conv2d_0b_{0}x{1}'.format(*kerenlsize))
        with tf.compat.v1.variable_scope('Branch_2'):
            tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, kerenlsize, scope='Conv2d_0b_{0}x{1}'.format(*kerenlsize))
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, kerenlsize, scope='Conv2d_0c_{0}x{1}'.format(*kerenlsize))
        with tf.compat.v1.variable_scope('Branch_3'):
            tower_conv3_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv3_1 = slim.conv2d(tower_conv3_0, 48, kerenlsize, scope='Conv2d_0b_{0}x{1}'.format(*kerenlsize))
            tower_conv3_2 = slim.conv2d(tower_conv3_1, 64, kerenlsize, scope='Conv2d_0c_{0}x{1}'.format(*kerenlsize))
            tower_conv3_3 = slim.conv2d(tower_conv3_2, 64, kerenlsize, scope='Conv2d_0d_{0}x{1}'.format(*kerenlsize))
        with tf.compat.v1.variable_scope('Branch_4'):
            tower_conv4_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv4_1 = slim.conv2d(tower_conv4_0, 48, kerenlsize, scope='Conv2d_0b_{0}x{1}'.format(*kerenlsize))
            tower_conv4_2 = slim.conv2d(tower_conv4_1, 48, kerenlsize, scope='Conv2d_0c_{0}x{1}'.format(*kerenlsize))
            tower_conv4_3 = slim.conv2d(tower_conv4_2, 64, kerenlsize, scope='Conv2d_0d_{0}x{1}'.format(*kerenlsize))
            tower_conv4_4 = slim.conv2d(tower_conv4_3, 64, kerenlsize, scope='Conv2d_0e_{0}x{1}'.format(*kerenlsize))
        mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_1, tower_conv2_2, tower_conv3_3, tower_conv4_4])
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None, activation_fn=None, scope='Conv2d_1x1')
        scaled_up = up * scale
        if activation_fn == tf.nn.relu6:
            # Use clip_by_value to simulate bandpass activation.
            scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

        net += scaled_up
        if activation_fn:
            net = activation_fn(net)
    return net


# In[36]:


def base_structure_1(inputs, output=1024, activation_fn=tf.nn.relu, scope=None):
    end_points = {}
    
    with tf.compat.v1.variable_scope(scope, 'InceptionResnet_Stock', [inputs]):
        with slim.arg_scope([slim.conv2d], stride=1, padding='SAME', activation_fn=activation_fn):
  
            net = conv2d_nearest(inputs, 32, 3, stride=(1,1), activation_fn=activation_fn, scope='Conv2d_1a_3x3')
            net = conv2d_nearest(net   , 32, 3, stride=(2,1), activation_fn=activation_fn, scope='Conv2d_2a_3x3')
            net = conv2d_nearest(net   , 32, 3, stride=(1,1), activation_fn=activation_fn, scope='Conv2d_3a_3x3')

            net = conv2d_nearest(net   , 64, 3, stride=(1,1), activation_fn=activation_fn, scope='Conv2d_1b_3x3')
            net = conv2d_nearest(net   , 64, 3, stride=(2,1), activation_fn=activation_fn, scope='Conv2d_2b_3x3')
            net = conv2d_nearest(net   , 64, 3, stride=(1,1), activation_fn=activation_fn, scope='Conv2d_3b_3x3')
            net = conv2d_nearest(net   , 80, 1, stride=(1,1), activation_fn=activation_fn, scope='Conv2d_4b_1x1')

            net = conv2d_nearest(net   ,192, 3, stride=(1,1), activation_fn=activation_fn, scope='Conv2d_1c_3x3')
            net = conv2d_nearest(net   ,192, 3, stride=(2,1), activation_fn=activation_fn, scope='Conv2d_2c_3x3')
            net = conv2d_nearest(net   ,192, 3, stride=(1,1), activation_fn=activation_fn, scope='Conv2d_3c_3x3')

            with tf.compat.v1.variable_scope('Mixed_1'):
                with tf.compat.v1.variable_scope('Branch_0'):
                    tower_conv    = conv2d_nearest(net          , 96, 1, activation_fn=activation_fn, scope='Conv2d_1x1')
                with tf.compat.v1.variable_scope('Branch_1'):
                    tower_conv1_0 = conv2d_nearest(net          , 48, 1, activation_fn=activation_fn, scope='Conv2d_0a_1x1')
                    tower_conv1_1 = conv2d_nearest(tower_conv1_0, 64, 5, activation_fn=activation_fn, scope='Conv2d_0b_5x5')
                with tf.compat.v1.variable_scope('Branch_2'):
                    tower_conv2_0 = conv2d_nearest(net          , 64, 1, activation_fn=activation_fn, scope='Conv2d_0a_1x1')
                    tower_conv2_1 = conv2d_nearest(tower_conv2_0, 96, 3, activation_fn=activation_fn, scope='Conv2d_0b_3x3')
                    tower_conv2_2 = conv2d_nearest(tower_conv2_1, 96, 3, activation_fn=activation_fn, scope='Conv2d_0c_3x3')
                with tf.compat.v1.variable_scope('Branch_3'):
                    tower_conv3_0 = conv2d_nearest(net          , 64, 1, activation_fn=activation_fn, scope='Conv2d_0a_1x1')
                    tower_conv3_1 = conv2d_nearest(tower_conv3_0, 64, 3, activation_fn=activation_fn, scope='Conv2d_0b_3x3')
                    tower_conv3_2 = conv2d_nearest(tower_conv3_1, 96, 3, activation_fn=activation_fn, scope='Conv2d_0c_3x3')
                    tower_conv3_3 = conv2d_nearest(tower_conv3_2, 96, 3, activation_fn=activation_fn, scope='Conv2d_0d_3x3')
                with tf.compat.v1.variable_scope('Branch_4'):
                    tower_pool    = avg_pool2d_nearest(net, 3, stride=1, scope='AvgPool_0a_3x3')
                    tower_pool_1  = conv2d_nearest(tower_pool   , 64, 1, activation_fn=activation_fn, scope='Conv2d_0b_1x1')
                net = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2, tower_conv3_3, tower_pool_1], 3)

            net = slim.repeat(net, 10, block319_nearest, scale=0.17, activation_fn=activation_fn, ratio=1)

            with tf.compat.v1.variable_scope('Mixed_2'):
                with tf.compat.v1.variable_scope('Branch_0'):
                    tower_conv    = conv2d_nearest(net          , 384, 3, activation_fn=activation_fn, scope='Conv2d_1a_3x3')
                with tf.compat.v1.variable_scope('Branch_1'):
                    tower_conv1_0 = conv2d_nearest(net          , 196, 1, activation_fn=activation_fn, scope='Conv2d_0a_1x1')
                    tower_conv1_1 = conv2d_nearest(tower_conv1_0, 256, 3, activation_fn=activation_fn, scope='Conv2d_0b_3x3')
                with tf.compat.v1.variable_scope('Branch_2'):
                    tower_conv2_0 = conv2d_nearest(net          , 256, 1, activation_fn=activation_fn, scope='Conv2d_0a_1x1')
                    tower_conv2_1 = conv2d_nearest(tower_conv2_0, 256, 3, activation_fn=activation_fn, scope='Conv2d_0b_3x3')
                    tower_conv2_2 = conv2d_nearest(tower_conv2_1, 384, 3, activation_fn=activation_fn, scope='Conv2d_0c_3x3')
                with tf.compat.v1.variable_scope('Branch_3'):
                    tower_conv3_0 = conv2d_nearest(net          , 256, 1, activation_fn=activation_fn, scope='Conv2d_0a_1x1')
                    tower_conv3_1 = conv2d_nearest(tower_conv3_0, 256, 3, activation_fn=activation_fn, scope='Conv2d_0b_3x3')
                    tower_conv3_2 = conv2d_nearest(tower_conv3_1, 384, 3, activation_fn=activation_fn, scope='Conv2d_0c_3x3')
                    tower_conv3_3 = conv2d_nearest(tower_conv3_2, 384, 3, activation_fn=activation_fn, scope='Conv2d_0d_3x3')
                with tf.compat.v1.variable_scope('Branch_4'):
                    tower_pool    = avg_pool2d_nearest(net, 3,  stride=1, scope='AvgPool_0a_3x3')
                    tower_pool_1  = conv2d_nearest(tower_pool   , 256, 1, activation_fn=activation_fn, scope='Conv2d_0b_1x1')
                net = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2, tower_conv3_3, tower_pool_1], 3)

            net = slim.repeat(net, 10, block319_nearest, scale=0.17, activation_fn=activation_fn, ratio=2)

#             with tf.compat.v1.variable_scope('Mixed_3'):
#                 with tf.compat.v1.variable_scope('Branch_0'):
#                     tower_conv    = conv2d_nearest(net          , 384, 3, activation_fn=activation_fn, scope='Conv2d_1a_3x3')
#                 with tf.compat.v1.variable_scope('Branch_1'):
#                     tower_conv1_0 = conv2d_nearest(net          , 196, 1, activation_fn=activation_fn, scope='Conv2d_0a_1x1')
#                     tower_conv1_1 = conv2d_nearest(tower_conv1_0, 256, 3, activation_fn=activation_fn, scope='Conv2d_0b_3x3')
#                 with tf.compat.v1.variable_scope('Branch_2'):
#                     tower_conv2_0 = conv2d_nearest(net          , 256, 1, activation_fn=activation_fn, scope='Conv2d_0a_1x1')
#                     tower_conv2_1 = conv2d_nearest(tower_conv2_0, 256, 3, activation_fn=activation_fn, scope='Conv2d_0b_3x3')
#                     tower_conv2_2 = conv2d_nearest(tower_conv2_1, 384, 3, activation_fn=activation_fn, scope='Conv2d_0c_3x3')
#                 with tf.compat.v1.variable_scope('Branch_3'):
#                     tower_conv3_0 = conv2d_nearest(net          , 256, 1, activation_fn=activation_fn, scope='Conv2d_0a_1x1')
#                     tower_conv3_1 = conv2d_nearest(tower_conv3_0, 256, 3, activation_fn=activation_fn, scope='Conv2d_0b_3x3')
#                     tower_conv3_2 = conv2d_nearest(tower_conv3_1, 384, 3, activation_fn=activation_fn, scope='Conv2d_0c_3x3')
#                     tower_conv3_3 = conv2d_nearest(tower_conv3_2, 384, 3, activation_fn=activation_fn, scope='Conv2d_0d_3x3')
#                 with tf.compat.v1.variable_scope('Branch_4'):
#                     tower_pool    = avg_pool2d_nearest(net, 3,  stride=1, scope='AvgPool_0a_3x3')
#                 net = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2, tower_conv3_3, tower_pool], 3)

#             net = slim.repeat(net, 10, block319_nearest, scale=0.17, activation_fn=activation_fn, ratio=3)
            net = conv2d_nearest(net, output, 1, scope='Conv2d_base_structure_1_1x1', activation_fn=tf.nn.relu)#tf.nn.tanh
            end_points['Conv2d_base_structure_1_1x1']=net
        return net, end_points

def base_structure_2(inputs, inputs2, num_classes, is_training=True, dropout_keep_prob=0.8, reuse=None, activation_fn=tf.nn.relu, scope="InceptionResnet_Stock"):
    with tf.compat.v1.variable_scope(scope, 'InceptionResnet_Stock', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            
            net, end_points = base_structure_1(inputs, scope=scope, activation_fn=activation_fn)
            
            with tf.compat.v1.variable_scope('Logits'):
                
#                 # global_avg_pool
#                 kernel_size = net.get_shape()[1:3]
#                 if kernel_size.is_fully_defined():
#                     global_avg_pool = slim.avg_pool2d(net, kernel_size, padding='VALID', scope='AvgPool_1a_txt')
#                 else:
#                     global_avg_pool = tf.reduce_mean(input_tensor=net, axis=[1, 2], keepdims=True, name='global_avg_pool')
#                 end_points['global_avg_pool'] = global_avg_pool
#                 #net = slim.flatten(global_avg_pool)

                # global_max_pool
                kernel_size = net.get_shape()[1:3]
                if kernel_size.is_fully_defined():
                    global_max_pool = slim.max_pool2d(net, kernel_size, padding='VALID', scope='MaxPool_1a_txt')
                else:
                    global_max_pool = tf.reduce_max(input_tensor=net, axis=[1, 2], keepdims=True, name='global_max_pool')
                end_points['global_max_pool'] = global_max_pool
                #net = slim.flatten(global_pool)
                
                # global_pool + location + newest days data
                print("global_max_pool=",global_max_pool.shape)
                print("net_1=",net.shape)
                net             = tf.transpose(net, (0,3,1,2))
                print("net_2=",net.shape)
                days_max        = tf.reduce_max(net, axis=3)
                print("days_max=",days_max.shape)
                global_pool_pos = 1/(1+slim.flatten(tf.argmax(days_max[:,:,::-1], axis=2)))
                print("global_pool_pos=",global_pool_pos.shape)
                #newest_days     = days_max[:,:,-5:]
                print("tf.shape(x)[0]=",tf.shape(days_max)[2])
                #newest_days     = tf.slice(days_max, [0, 0, days_max.get_shape()[2]-5], [1, 2024, 5])
                newest_days     = tf.slice(days_max, [0, 0, tf.shape(days_max)[2]-5], [1, 1024, 5])
                print("newest_days_1=",newest_days.shape)
                newest_days     = slim.flatten(newest_days)
                #newest_days     = tf.reshape(days_max, [1, tf.shape(newest_days)[1]*tf.shape(newest_days)[2]])
                print("newest_days_2=",newest_days.shape)
                net = tf.concat([tf.cast(slim.flatten(global_max_pool), tf.float32), tf.cast(global_pool_pos, tf.float32), tf.cast(newest_days, tf.float32), inputs2],1)
                print("net=",slim.flatten(net).shape)
                net    = slim.flatten(net)
                print("net_final=",net.shape)
                end_points['global_pool_with_others'] = net
                
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='Dropout')
                end_points['PreLogitsFlatten'] = net
                
                logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='Logits')
                end_points['Logits'] = logits
                end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')
    return logits, end_points


# # Data Preparation

# In[2]:


import fixDay_preprocess_3 as preprocess

# In[47]:
preprocess.preProcess(fix_days, tp, sl)

train_x, train_y = preprocess.grabData(label=target_label, startT=trainDays[0], endT=trainDays[1])
train_x, train_y = train_x[66:], train_y[66:]
val_x  , val_y   = preprocess.grabData(label=target_label, startT=valDays[0]  , endT=valDays[1])
test_x , test_y  = preprocess.grabData(label=target_label, startT=testDays[0] , endT=testDays[1])


# # Start Train Flow

# In[ ]:


import numpy as np

main_graph = tf.Graph()
sess = tf.Session(graph=main_graph)

with main_graph.as_default():
    node       = tf.placeholder(shape=[1, None, parameterN, 1], dtype=tf.float32)
    rangeV     = tf.placeholder(shape=(1, 1), dtype=tf.float32)
    ans        = tf.placeholder(shape=(1, 3), dtype=tf.float32)
    loss_ratio = tf.placeholder_with_default(1.0  , shape=(), name="loss_ratio")
    #is_train = tf.placeholder(shape=[1, ], dtype=tf.bool)
    is_train   = tf.placeholder_with_default(False, shape=(), name="is_training")
    
    logits, end_points = base_structure_2(node, rangeV, 3, is_training=is_train, dropout_keep_prob=0.8, activation_fn=tf.nn.leaky_relu)
    loss     = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ans, logits=logits))*loss_ratio
    
    opt      = tf.train.AdamOptimizer(5e-6, beta1=0.5, beta2=0.999)
    update   = opt.minimize(loss) 
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()    
    
tf.summary.FileWriter("./detect", graph=tf.get_default_graph())
sess.run(init)
#sess.run(tf.global_variables_initializer())
#sess.run(tf.local_variables_initializer())


# In[ ]:
from callbacks import *

model_name = "Stock_InceptionResnet"

model_dict = {
    'model_name' : model_name,
    'checkpoint' : Model_checkpoint(os.path.join(savepath, subsavepath, model_name), recordStep=recordStep, save_best_only=True , lastTime=epochs),
    'train_batch_log' : History(['loss']),
    'val_batch_log' : History(['loss']),
    'history' : {
        'train_loss':[],
        'val_loss':[]
    }
}

callback_dict = {
    'on_session_begin':[], # start of a session
    'on_batch_begin':[], # start of a training batch
    'on_batch_end':[], # end of a training batch
    'on_epoch_begin':[], # start of a epoch
    'on_epoch_end':[
        model_dict['checkpoint'],
        #Model_checkpoint(os.path.join(savepath, subsavepath, model_name), recordStep=recordStep, save_best_only=False, lastTime=epochs)
    ], # end of a epoch
    'on_session_end':[] # end of a session
}
callback_manager = Run_collected_functions(callback_dict)


# In[ ]:
lossRecord  = np.inf
loss_ratio0 = np.float32([1.,1.2,1.2])
val_number  = np.float32([np.sum(val_y==[1,0,0]), np.sum(val_y==[0,1,0]), np.sum(val_y==[0,0,1])])
for epoch in range(epochs):
    
    ### train ###
    for batch in range(batchs):
        label = np.random.randint(3)
        y = np.array([[0,0,0]])
        y[0, label]=1
        
        x = np.where(np.all(train_y==y[0],axis=1))[0]
        x = np.random.choice(x, 1)[0]
        x1, x2 = train_x[x]

        # 執行 loss & update (train)
        _, train_loss, train_logits = sess.run([update, loss, logits], feed_dict={node      : x1[np.newaxis, :,parameter_Order, np.newaxis], 
                                                                                  rangeV    : np.float32([x2])[np.newaxis, :],
                                                                                  ans       : y, 
                                                                                  loss_ratio: loss_ratio0[label], 
                                                                                  is_train  : True})
        model_dict['train_batch_log'].push({'loss':train_loss})
        #print("train_logits=",train_logits)
#         print("global_pool_with_others=",np.sum(np.isnan(train_end_points['global_pool_with_others'])))
#         print("PreLogitsFlatten=",np.sum(np.isnan(train_end_points['PreLogitsFlatten'])))
#         print("Logits=",np.sum(np.isnan(train_end_points['Logits'])))
#         print("Logits=",train_end_points['Logits'])
#         print("Predictions=",np.sum(np.isnan(train_end_points['Predictions'])))
                
    model_dict['history']['train_loss'].append(model_dict['train_batch_log'].avg_value('loss'))
    model_dict['train_batch_log'].reset()

    ### val ###
    predlist, anslist = [], []
    val_total = np.sum(val_number)/3.
    for (x1,x2), y in zip(val_x, val_y):
        ratio = np.argmax(y)
        val_loss, val_end_points = sess.run([loss, end_points], feed_dict={node      : x1[np.newaxis, :,parameter_Order, np.newaxis], 
                                                                           rangeV    : np.float32([x2])[np.newaxis, :],
                                                                           ans       : y[np.newaxis,:], 
                                                                           loss_ratio: loss_ratio0[ratio]/val_number[ratio]*val_total, 
                                                                           is_train  : False})
        model_dict['val_batch_log'].push({'loss':val_loss})
        
        pred = np.argmax(val_end_points['Predictions'], axis=1)
        true = np.argmax(y[np.newaxis,:], axis=1)
        predlist.append(pred[0])
        anslist.append(true[0])
    
    predlist = np.array(predlist)
    anslist  = np.array(anslist)
    
    record = ""
    print(" ______________________")
    print(" T\P |  0  |  1  |  2  |")
    print(" -----------------------")
    record = record + " ______________________\n T\\P |  0  |  1  |  2  |\n -----------------------\n"
    for i in range(3):
        mask = anslist==i
        preN = []
        for i2 in range(3):
            preN.append(np.sum(predlist[mask]==i2))
        print("{0:^5}|{1:>5}|{2:>5}|{3:>5}|".format(i, preN[0], preN[1], preN[2]))
        print(" -----------------------")
        record = record + "{0:^5}|{1:>5}|{2:>5}|{3:>5}|\n -----------------------\n".format(i, preN[0], preN[1], preN[2])

    model_dict['history']['val_loss'].append(model_dict['val_batch_log'].avg_value('loss'))
    model_dict['val_batch_log'].reset()

    ### callback ###
    print('Epoch: {}/{}'.format(epoch,epochs))
    print('train_loss: {:.3f}'.format(model_dict['history']['train_loss'][-1]))
    print('val_loss: {:.3f}'.format(model_dict['history']['val_loss'][-1]))

    callback_manager.run_on_epoch_end(val_loss = model_dict['history']['val_loss'][-1],
                                      sess = sess,
                                      saver = saver,
                                      nth_epoch = epoch)
    print('############################')
    
    record     = record + 'Epoch: {}/{}\n'.format(epoch,epochs) + \
                          'train_loss: {:.3f}\n'.format(model_dict['history']['train_loss'][-1]) + \
                          'val_loss: {:.3f}\n'.format(model_dict['history']['val_loss'][-1]) + \
                          '############################\n'    
    
    if lossRecord > model_dict['history']['val_loss'][-1]:
        lossRecord = model_dict['history']['val_loss'][-1]
        with open(os.path.join(savepath, subsavepath,"BestRecord.txt"),"a+") as f:
            f.write(record)
            
    if epoch >= epochs-1:
        with open(os.path.join(savepath, subsavepath,"BestRecord.txt"),"a+") as f:
            f.write(record)       
#     if epoch%recordStep==0:
#         with open(os.path.join(savepath, subsavepath, model_name, "BestRecord.txt"),"a+") as f:
#             f.write(record)

# In[ ]:




