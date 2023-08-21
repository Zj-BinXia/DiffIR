import tensorflow as tf


def extract_image_patches(image_batch, patch_size,patch_stride):	
	patches = tf.extract_image_patches(images =image_batch,ksizes=[1,patch_size,patch_size,1],rates=[1,1,1,1],strides=[1,patch_stride,patch_stride,1],padding='VALID')
	patches_shape = patches.get_shape().as_list()
	return tf.reshape(patches,[-1,patch_size,patch_size,3])#, patches_shape[1]*patches_shape[2] # NOTE: assuming 3 channels


def conv_init(name,input_channels, filter_height, filter_width, num_filters, groups=1):
  weights = get_scope_variable(name, 'weights', shape=[filter_height, filter_width, input_channels/groups, num_filters], trainable=False)
  biases = get_scope_variable(name, 'biases', shape = [num_filters],trainable=False)


def fc_init(name, num_in, num_out):
  weights = get_scope_variable(name, 'weights', shape=[num_in, num_out], trainable=False)
  biases = get_scope_variable(name, 'biases', shape=[num_out], trainable=False)


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', relu=True):
  input_channels = int(x.get_shape().as_list()[3])  
  convolve = lambda i, k: tf.nn.conv2d(i, k, strides = [1, stride_y, stride_x, 1], padding = padding)    
  weights = get_scope_variable(name, 'weights', shape=[filter_height, filter_width, input_channels, num_filters])
  biases = get_scope_variable(name, 'biases', shape = [num_filters])
  conv = convolve(x, weights)      
  bias_val = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))
  if relu == True:
    relu = tf.nn.relu(bias_val, name = name)       
    return relu
  else:
    return bias_val
  

def fc(x, num_in, num_out, name, relu = True):
	weights = get_scope_variable(name, 'weights', shape=[num_in, num_out])
	biases = get_scope_variable(name, 'biases', shape=[num_out])  
	act = tf.nn.xw_plus_b(x, weights, biases, name=name)  
	if relu == True:
		relu = tf.nn.relu(act)      
		return relu
	else:
		return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
  return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1], strides = [1, stride_y, stride_x, 1], padding = padding, name = name)


def dropout(x, keep_prob):
  return tf.nn.dropout(x, keep_prob)


def get_scope_variable(scope_name, var, shape=None, initialvals=None,trainable=False):
	with tf.variable_scope(scope_name,reuse=tf.AUTO_REUSE) as scope:
		v = tf.get_variable(var,shape,dtype=tf.float32, initializer=initialvals,trainable=trainable)
	return v


