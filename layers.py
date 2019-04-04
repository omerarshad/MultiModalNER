

import numpy as np
import tensorflow as tf
import os



import os
from collections import defaultdict
import numpy as np
import tensorflow as tf
from functools import reduce
from operator import mul
import collections

import math


from functools import reduce

SEED = 86
np.random.seed(SEED)



VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER  

def embedding_layer(input_placeholder=None,
                    token_embedding_matrix=None,
                    n_tokens=None,
                    token_embedding_dim=None,
                    name=None,
                    trainable=True):
    if token_embedding_matrix is not None:
        tok_mat = token_embedding_matrix
        if trainable:
            
            print("Embeddings paramenters are set to Trainable")
            Warning('Matrix of embeddings is passed to the embedding_layer, '
                    'possibly there is a pre-trained embedding matrix. '
                    'Embeddings paramenters are set to Trainable!')
    else:
        tok_mat = np.random.randn(n_tokens, token_embedding_dim).astype(np.float32) / np.sqrt(token_embedding_dim)
    tok_emb_mat = tf.Variable(tok_mat, name=name, trainable=trainable)
    embeddings = tf.nn.embedding_lookup(tok_emb_mat, input_placeholder)
#     embeddings = add_timing_signal(embeddings)
    return embeddings

def layer_normalize(inputs, epsilon=1e-8, scope=None):
    with tf.variable_scope(scope or "layer_norm"):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** 0.5)
        outputs = tf.add(tf.multiply(gamma, normalized), beta)
    return outputs



def character_embedding_network(char_placeholder, n_characters, char_embedding_dim, filter_width=7):
    char_emb_mat = np.random.randn(n_characters, char_embedding_dim).astype(np.float32) / np.sqrt(char_embedding_dim)
    char_emb_var = tf.Variable(char_emb_mat, trainable=True)
    #char_emb_var = tf.nn.dropout(char_emb_mat,0.8,noise_shape=[n_characters,1])
    with tf.variable_scope('Char_Emb_Network'):
        # Character embedding layer
        c_emb = tf.nn.embedding_lookup(char_emb_var, char_placeholder)

        # Character embedding network
        char_conv = tf.layers.conv2d(c_emb, char_embedding_dim, (1, filter_width), padding='same', name='char_conv')
        print("char_conv ",char_conv.shape)
        char_emb = tf.reduce_max(char_conv, axis=2)
        print("char_emb ",char_emb.shape)
    return char_emb

  

def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat  
  

def reconstruct(tensor, ref, keep, dim_reduced_keep=None):
    dim_reduced_keep = dim_reduced_keep or keep

    ref_shape = ref.get_shape().as_list() # original shape
    tensor_shape = tensor.get_shape().as_list() # current shape
    ref_stop = len(ref_shape) - keep # flatten dims list
    tensor_start = len(tensor_shape) - dim_reduced_keep  # start
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)] #
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))] #
    # pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
    # keep_shape = tensor.get_shape().as_list()[-keep:]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out


def exp_mask_for_high_rank(val, val_mask, name=None):
    val_mask = tf.expand_dims(val_mask, -1)
    return tf.add(val, (1 - tf.cast(val_mask, tf.float32)) * VERY_NEGATIVE_NUMBER,
                  name=name or 'exp_mask_for_high_rank')
def mask_for_high_rank(val, val_mask, name=None):
    val_mask = tf.expand_dims(val_mask, -1)
    return tf.multiply(val, tf.cast(val_mask, tf.float32), name=name or 'mask_for_high_rank')

     
    
def bn_dense_layer(input_tensor, hn, bias, bias_start=0.0, scope=None,
                   activation='relu', enable_bn=True,
                   wd=0., keep_prob=1.0, is_train=None):
    if is_train is None:
        is_train = False

    # activation
    if activation == 'linear':
        activation_func = tf.identity
    elif activation == "tanh":
        activation_func = tf.nn.tanh
    elif activation == 'relu':
        activation_func = tf.nn.relu
    elif activation == 'elu':
        activation_func = tf.nn.elu
    elif activation == 'selu':
        activation_func = selu
    else:
        raise AttributeError('no activation function named as %s' % activation)

    with tf.variable_scope(scope or 'bn_dense_layer'):
        linear_map = linear(input_tensor, hn, bias, bias_start, 'linear_map',
                            False, wd, keep_prob, is_train)
        if enable_bn:
            linear_map = tf.contrib.layers.batch_norm(
                linear_map, center=True, scale=True, is_training=is_train, scope='bn')
        return activation_func(linear_map)
      
      
      
def linear(args, output_size, bias, bias_start=0.0, scope=None, squeeze=False, wd=0.0, input_keep_prob=1.0,
           is_train=None):
    if args is None or (isinstance(args, (tuple, list)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (tuple, list)):
        args = [args]

    flat_args = [flatten(arg, 1) for arg in args] # for dense layer [(-1, d)]
    if input_keep_prob < 1.0:
        assert is_train is not None
        flat_args = [tf.cond(is_train, lambda: tf.nn.dropout(arg, input_keep_prob), lambda: arg)# for dense layer [(-1, d)]
                     for arg in flat_args]
    flat_out = _linear(flat_args, output_size, bias, bias_start=bias_start, scope=scope) # dense
    out = reconstruct(flat_out, args[0], 1) # ()
    if squeeze:
        out = tf.squeeze(out, [len(args[0].get_shape().as_list())-1])

    if wd:
        add_reg_without_bias()

    return out


def add_reg_without_bias(scope=None):
    scope = scope or tf.get_variable_scope().name
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    counter = 0
    for var in variables:
        if len(var.get_shape().as_list()) <= 1: continue
        tf.add_to_collection('reg_vars', var)
        counter += 1
    return counter  
  

def _linear(xs,output_size,bias,bias_start=0., scope=None):
    with tf.variable_scope(scope or 'linear_layer'):
        x = tf.concat(xs,-1)
        input_size = x.get_shape()[-1]
        W = tf.get_variable('W', shape=[input_size,output_size],dtype=tf.float32,
                            )
        if bias:
            bias = tf.get_variable('bias', shape=[output_size],dtype=tf.float32,
                                   initializer=tf.constant_initializer(bias_start))
            out = tf.matmul(x, W) + bias
        else:
            out = tf.matmul(x, W)
        return out


def dropout(x, keep_prob, is_train, noise_shape=None, seed=None, name=None):
    with tf.name_scope(name or "dropout"):
        assert is_train is not None
        if keep_prob < 1.0:
            d = tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed)
            out = tf.cond(is_train, lambda: d, lambda: x)
            return out
        return x