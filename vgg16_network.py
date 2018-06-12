
# coding: utf-8

# In[1]:


import tensorflow as tf
import scipy.io  
import numpy as np
from tensorflow.python import pywrap_tensorflow 


# In[2]:


VERBOSE = 0


# In[19]:


def get_weights(vgg_para, layer_num,regularizer,layer_name=None,trainEn=True,checkpoint=False):
    if checkpoint:
        weights = vgg_para.get_tensor(layer_name)
    else:
        weights = vgg_para[layer_num][0][0][2][0][0]
    W = tf.Variable(initial_value=weights,trainable=trainEn,name=layer_name)
    if regularizer!=None:
        tf.add_to_collection('losses',regularizer(W))
    if VERBOSE : print("layer name is:",vgg_para[layer_num][0][0][0],"layer weights shape :",W.shape)
    return W

def get_bias(vgg_para, layer_num,layer_name=None,trainEn=True,checkpoint=False):
    if checkpoint:
        bias = vgg_para.get_tensor(layer_name)
    else:
        bias = vgg_para[layer_num][0][0][2][0][1]
    b = tf.Variable(initial_value=np.reshape(bias, (bias.size)),trainable=trainEn,name=layer_name)
    if VERBOSE : print("layer name is:",vgg_para[layer_num][0][0][0],layer_num,"layer bias shape :",b.shape)
    return b

def conv_layer(layer_name, layer_input, W):
    conv = tf.nn.conv2d(layer_input, W, strides=[1, 1, 1, 1], padding='SAME')
    if VERBOSE:
        print('--{} | shape={} | layer_input={} | weights_shape={}'.format(layer_name, conv.shape,layer_input.shape, W.shape))
    return conv

def relu_layer(layer_name, layer_input, b):
    relu = tf.nn.relu(layer_input + b)
    if VERBOSE: 
        print('--{} | shape={} | bias_shape={}'.format(layer_name, relu.shape, b.shape))
    return relu

def pool_layer(layer_name, layer_input):
    pool = tf.nn.max_pool(layer_input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    if VERBOSE: 
        print('--{}   | shape={}'.format(layer_name, pool.get_shape()))
    return pool


# In[20]:


def VGG16(img,keep_prob,regularizer=None,checkpoint=False,checkpoint_dir=None,ckt_name=None):   
    if VERBOSE: print('loading model weights...')
    if checkpoint:
        print("continue train",checkpoint_dir,ckt_name)
        vgg_layers = pywrap_tensorflow.NewCheckpointReader(checkpoint_dir + ckt_name)
    else:
        vgg_wrap = scipy.io.loadmat("imagenet-vgg-verydeep-16.mat")
        vgg_layers = vgg_wrap['layers'][0]
    
    if VERBOSE: print('constructing layers...')
    if VERBOSE: print('LAYER GROUP 1')
    
    ####################VGG-16################
    
    x = conv_layer('conv1_1', img, W=get_weights(vgg_layers, 0,regularizer,layer_name='vggconv1_1',trainEn=True,checkpoint=checkpoint))
    x = relu_layer('relu1_1', x, b=get_bias(vgg_layers, 0,layer_name='vggrelu1_1',trainEn=True,checkpoint=checkpoint))

    x = conv_layer('conv1_2', x, W=get_weights(vgg_layers, 2,regularizer,layer_name='vggconv1_2',trainEn=True,checkpoint=checkpoint))
    x = relu_layer('relu1_2', x, b=get_bias(vgg_layers, 2,layer_name='vggrelu1_2',trainEn=True,checkpoint=checkpoint))

    x   = pool_layer('pool1', x)

    if VERBOSE: print('LAYER GROUP 2')  
    x = conv_layer('conv2_1', x, W=get_weights(vgg_layers, 5,regularizer,layer_name='vggconv2_1',trainEn=True,checkpoint=checkpoint))
    x = relu_layer('relu2_1', x, b=get_bias(vgg_layers, 5,layer_name='vggrelu2_1',trainEn=True,checkpoint=checkpoint))

    x = conv_layer('conv2_2', x, W=get_weights(vgg_layers, 7,regularizer,layer_name='vggconv2_2',trainEn=True,checkpoint=checkpoint))
    x = relu_layer('relu2_2', x, b=get_bias(vgg_layers, 7,layer_name='vggrelu2_2',trainEn=True,checkpoint=checkpoint))

    x   = pool_layer('pool2', x)

    if VERBOSE: print('LAYER GROUP 3')
    x = conv_layer('conv3_1', x, W=get_weights(vgg_layers, 10,regularizer,layer_name='vggconv3_1',trainEn=True,checkpoint=checkpoint))
    x = relu_layer('relu3_1', x, b=get_bias(vgg_layers, 10,layer_name='vggrelu3_1',trainEn=True,checkpoint=checkpoint))

    x = conv_layer('conv3_2', x, W=get_weights(vgg_layers, 12,regularizer,layer_name='vggconv3_2',trainEn=True,checkpoint=checkpoint))
    x = relu_layer('relu3_2', x, b=get_bias(vgg_layers, 12,layer_name='vggrelu3_2',trainEn=True,checkpoint=checkpoint))

    x = conv_layer('conv3_3', x, W=get_weights(vgg_layers, 14,regularizer,layer_name='vggconv3_3',trainEn=True,checkpoint=checkpoint))
    x = relu_layer('relu3_3', x, b=get_bias(vgg_layers, 14,layer_name='vggrelu3_3',trainEn=True,checkpoint=checkpoint))

    x   = pool_layer('pool3', x)

    if VERBOSE: print('LAYER GROUP 4')
    x = conv_layer('conv4_1', x, W=get_weights(vgg_layers, 17,regularizer,layer_name='vggconv4_1',trainEn=True,checkpoint=checkpoint))
    x = relu_layer('relu4_1', x, b=get_bias(vgg_layers, 17,layer_name='vggrelu4_1',trainEn=True,checkpoint=checkpoint))

    x = conv_layer('conv4_2', x, W=get_weights(vgg_layers, 19,regularizer,layer_name='vggconv4_2',trainEn=True,checkpoint=checkpoint))
    x = relu_layer('relu4_2', x, b=get_bias(vgg_layers, 19,layer_name='vggrelu4_2',trainEn=True,checkpoint=checkpoint))

    x = conv_layer('conv4_3', x, W=get_weights(vgg_layers, 21,regularizer,layer_name='vggconv4_3',trainEn=True,checkpoint=checkpoint))
    x = relu_layer('relu4_3', x, b=get_bias(vgg_layers, 21,layer_name='vggrelu4_3',trainEn=True,checkpoint=checkpoint))
    x   = pool_layer('pool4', x)

    if VERBOSE: print('LAYER GROUP 5')
    x = conv_layer('conv5_1', x, W=get_weights(vgg_layers, 24,regularizer,layer_name='vggconv5_1',trainEn=True,checkpoint=checkpoint))
    x = relu_layer('relu5_1', x, b=get_bias(vgg_layers, 24,layer_name='vggrelu5_1',trainEn=True,checkpoint=checkpoint))

    x = conv_layer('conv5_2', x, W=get_weights(vgg_layers, 26,regularizer,layer_name='vggconv5_2',trainEn=True,checkpoint=checkpoint))
    x = relu_layer('relu5_2', x, b=get_bias(vgg_layers, 26,layer_name='vggrelu5_2',trainEn=True,checkpoint=checkpoint))

    x = conv_layer('conv5_3', x, W=get_weights(vgg_layers, 28,regularizer,layer_name='vggconv5_3',trainEn=True,checkpoint=checkpoint))
    x = relu_layer('relu5_3', x, b=get_bias(vgg_layers, 28,layer_name='vggrelu5_3',trainEn=True,checkpoint=checkpoint))
    x = pool_layer('pool5', x)
    feature = x
    
    ####################  FC ################
    pool_node1 = 7
    pool_node2 = 7
    pool_node3 = 512
    FC_NODE = 1024
    nodes = pool_node1*pool_node2*pool_node3
    fc_input = tf.reshape(feature, [-1, nodes])
    
    if checkpoint:
        fc1_weights = get_weights(vgg_layers, 28,regularizer,layer_name='fc1_weights',trainEn=True,checkpoint=checkpoint)
        fc2_weights = get_weights(vgg_layers, 28,regularizer,layer_name='fc2_weights',trainEn=True,checkpoint=checkpoint)
        fc_cls_weights = get_weights(vgg_layers, 28,regularizer,layer_name='fc_cls_weights',trainEn=True,checkpoint=checkpoint)

        fc1_biases = get_bias(vgg_layers, 28,layer_name='fc_bias1',trainEn=True,checkpoint=checkpoint)
        fc2_biases = get_bias(vgg_layers, 28,layer_name='fc_bias2',trainEn=True,checkpoint=checkpoint)
        fc_cls_biases = get_bias(vgg_layers, 28,layer_name='bias_roi_cls',trainEn=True,checkpoint=checkpoint)
    else:
        fc1_weights = tf.Variable(initial_value=tf.truncated_normal(shape=[nodes, FC_NODE],stddev=0.01,dtype=tf.float32),
                                  name="fc1_weights",trainable=True)
        fc2_weights = tf.Variable(initial_value=tf.truncated_normal(shape=[FC_NODE, FC_NODE],stddev=0.01,dtype=tf.float32),
                                  name="fc2_weights",trainable=True)
        fc_cls_weights = tf.Variable(initial_value=tf.truncated_normal(shape=[FC_NODE, 10],stddev=0.01,dtype=tf.float32),
                                     name="fc_cls_weights",trainable=True)
        fc1_biases = tf.Variable(initial_value=tf.constant(0.01,shape=[FC_NODE],dtype=tf.float32),name="fc_bias1",trainable=True)
        fc2_biases = tf.Variable(initial_value=tf.constant(0.01,shape=[FC_NODE],dtype=tf.float32),name="fc_bias2",trainable=True)
        fc_cls_biases = tf.Variable(initial_value=tf.constant(0.01,shape=[10],dtype=tf.float32),name="bias_roi_cls",trainable=True)
    
    if regularizer!=None:
        tf.add_to_collection('losses',regularizer(fc1_weights))
        tf.add_to_collection('losses',regularizer(fc2_weights))
        tf.add_to_collection('losses',regularizer(fc_cls_weights))

    fc1 = tf.nn.relu(tf.matmul(fc_input, fc1_weights) + fc1_biases)
    fc1 = tf.nn.dropout(fc1, keep_prob)
    
    fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
    fc2 = tf.nn.dropout(fc2, keep_prob)
        
    fc_cls_10 = tf.nn.relu(tf.matmul(fc2, fc_cls_weights) + fc_cls_biases)
    if VERBOSE: print('--fc input shape',fc_input.shape,"fc1_weights",fc1_weights.shape,"fc1_biases",fc1_biases.shape)
    if VERBOSE: print('--fc1 shape',fc1.shape,"fc2_weights",fc2_weights.shape,"fc2_biases",fc1_biases.shape)
    if VERBOSE: print('--fc2 shape',fc2.shape,"fc_cls_weights,",fc_cls_weights.shape,"fc_cls_biases",fc_cls_biases.shape)
    if VERBOSE: print('--fc_cls_10 shape',fc_cls_10.shape)
    return fc_cls_10


# In[21]:


def get_test_weights(vgg_para,layer_name=None):
    weights = vgg_para.get_tensor(layer_name)
    W = tf.constant(weights,name='layer_name',dtype = tf.float32)
    if VERBOSE : print("layer name is:",layer_name,"layer weights shape :",W.shape)
    return W

def get_test_bias(vgg_para,layer_name=None):
    bias = vgg_para.get_tensor(layer_name)
    b = tf.constant(bias,name=layer_name,dtype=tf.float32)
    if VERBOSE : print("layer name is:",layer_name,"layer bias shape :",b.shape)
    return b


# In[22]:


def VGG16_test(img,checkpoint_dir,ckt_name): 
    vgg_layers = pywrap_tensorflow.NewCheckpointReader(checkpoint_dir + ckt_name)
    ####################VGG-16################
    
    x = conv_layer('conv1_1', img, W=get_test_weights(vgg_layers,layer_name='vggconv1_1'))
    x = relu_layer('relu1_1', x, b=get_test_bias(vgg_layers, layer_name='vggrelu1_1'))

    x = conv_layer('conv1_2', x, W=get_test_weights(vgg_layers,layer_name='vggconv1_2'))
    x = relu_layer('relu1_2', x, b=get_test_bias(vgg_layers,layer_name='vggrelu1_2'))

    x   = pool_layer('pool1', x)

    if VERBOSE: print('LAYER GROUP 2')  
    x = conv_layer('conv2_1', x, W=get_test_weights(vgg_layers, layer_name='vggconv2_1'))
    x = relu_layer('relu2_1', x, b=get_test_bias(vgg_layers, layer_name='vggrelu2_1'))

    x = conv_layer('conv2_2', x, W=get_test_weights(vgg_layers,layer_name='vggconv2_2'))
    x = relu_layer('relu2_2', x, b=get_test_bias(vgg_layers, layer_name='vggrelu2_2'))

    x   = pool_layer('pool2', x)

    if VERBOSE: print('LAYER GROUP 3')
    x = conv_layer('conv3_1', x, W=get_test_weights(vgg_layers,layer_name='vggconv3_1'))
    x = relu_layer('relu3_1', x, b=get_test_bias(vgg_layers, layer_name='vggrelu3_1'))

    x = conv_layer('conv3_2', x, W=get_test_weights(vgg_layers,layer_name='vggconv3_2'))
    x = relu_layer('relu3_2', x, b=get_test_bias(vgg_layers,layer_name='vggrelu3_2'))

    x = conv_layer('conv3_3', x, W=get_test_weights(vgg_layers,layer_name='vggconv3_3'))
    x = relu_layer('relu3_3', x, b=get_test_bias(vgg_layers, layer_name='vggrelu3_3'))

    x   = pool_layer('pool3', x)

    if VERBOSE: print('LAYER GROUP 4')
    x = conv_layer('conv4_1', x, W=get_test_weights(vgg_layers,layer_name='vggconv4_1'))
    x = relu_layer('relu4_1', x, b=get_test_bias(vgg_layers,layer_name='vggrelu4_1'))

    x = conv_layer('conv4_2', x, W=get_test_weights(vgg_layers,layer_name='vggconv4_2'))
    x = relu_layer('relu4_2', x, b=get_test_bias(vgg_layers,layer_name='vggrelu4_2'))

    x = conv_layer('conv4_3', x, W=get_test_weights(vgg_layers,layer_name='vggconv4_3'))
    x = relu_layer('relu4_3', x, b=get_test_bias(vgg_layers,layer_name='vggrelu4_3'))
    x   = pool_layer('pool4', x)

    if VERBOSE: print('LAYER GROUP 5')
    x = conv_layer('conv5_1', x, W=get_test_weights(vgg_layers,layer_name='vggconv5_1'))
    x = relu_layer('relu5_1', x, b=get_test_bias(vgg_layers,layer_name='vggrelu5_1'))

    x = conv_layer('conv5_2', x, W=get_test_weights(vgg_layers,layer_name='vggconv5_2'))
    x = relu_layer('relu5_2', x, b=get_test_bias(vgg_layers,layer_name='vggrelu5_2'))

    x = conv_layer('conv5_3', x, W=get_test_weights(vgg_layers,layer_name='vggconv5_3'))
    x = relu_layer('relu5_3', x, b=get_test_bias(vgg_layers,layer_name='vggrelu5_3'))
    x = pool_layer('pool5', x)
    
    fc_input = tf.reshape(x, [1, 7*7*512])
    
    fc1_weights = get_test_weights(vgg_layers,layer_name='fc1_weights')
    fc2_weights = get_test_weights(vgg_layers,layer_name='fc2_weights')
    fc_cls_weights = get_test_weights(vgg_layers,layer_name='fc_cls_weights')

    fc1_biases = get_test_bias(vgg_layers,layer_name='fc_bias2')
    fc2_biases = get_test_bias(vgg_layers,layer_name='fc_bias2')
    fc_cls_biases = get_test_bias(vgg_layers,layer_name='bias_roi_cls')
    
    fc1 = tf.nn.relu(tf.matmul(fc_input, fc1_weights) + fc1_biases)    
    fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
    fc_cls_10 = tf.nn.relu(tf.matmul(fc2, fc_cls_weights) + fc_cls_biases)
    return fc_cls_10

