import tensorflow as tf

def conv_layer(x, filter_shape, stride, initializer=None):
    if initializer is None:
        initializer = tf.contrib.layers.xavier_initializer()
    filters = tf.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=initializer,
        trainable=True)
    return tf.nn.conv2d(x, filters, [1, stride, stride, 1], padding='SAME')


def dilated_conv_layer(x, filter_shape, dilation, initializer=None):
    if initializer is None:
        initializer = tf.contrib.layers.xavier_initializer()
    filters = tf.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=initializer,
        trainable=True)
    return tf.nn.atrous_conv2d(x, filters, dilation, padding='SAME')


def deconv_layer(x, filter_shape, output_shape, stride, initializer=None):
    if initializer is None:
        initializer = tf.contrib.layers.xavier_initializer()
    filters = tf.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=initializer,
        trainable=True)
    return tf.nn.conv2d_transpose(x, filters, output_shape, [1, stride, stride, 1])


def batch_normalize(x, is_training, decay=0.99, epsilon=1e-5,
                   initializer_beta=None, initializer_scale=None, initializer_mean=None, initializer_var=None):
    def bn_train():
        batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon)

    def bn_inference():
        return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, scale, epsilon)

    if initializer_beta is None:
        initializer_beta = tf.truncated_normal_initializer(stddev=0.0)
    if initializer_scale is None:
        initializer_scale = tf.truncated_normal_initializer(stddev=0.1)
    if initializer_mean is None:
        initializer_mean = tf.constant_initializer(0.0)
    if initializer_var is None:
        initializer_var = tf.constant_initializer(1.0)
        
    dim = x.get_shape().as_list()[-1]
    beta = tf.get_variable(
        name='beta',
        shape=[dim],
        dtype=tf.float32,
        initializer=initializer_beta,
        trainable=True)
    scale = tf.get_variable(
        name='scale',
        shape=[dim],
        dtype=tf.float32,
        initializer=initializer_scale,
        trainable=True)
    pop_mean = tf.get_variable(
        name='pop_mean',
        shape=[dim],
        dtype=tf.float32,
        initializer=initializer_mean,
        trainable=False)
    pop_var = tf.get_variable(
        name='pop_var',
        shape=[dim],
        dtype=tf.float32,
        initializer=initializer_var,
        trainable=False)

    return tf.cond(is_training, bn_train, bn_inference)


def flatten_layer(x):
    input_shape = x.get_shape().as_list()
    dim = input_shape[1] * input_shape[2] * input_shape[3]
    transposed = tf.transpose(x, (0, 3, 1, 2))
    return tf.reshape(transposed, [-1, dim])


def full_connection_layer(x, out_dim, initializer_w=None, initializer_b=None):
    if initializer_w is None:
        initializer_w = tf.truncated_normal_initializer(stddev=0.1)
    if initializer_b is None:
        initializer_b = tf.constant_initializer(0.0)
        
    in_dim = x.get_shape().as_list()[-1]
    W = tf.get_variable(
        name='weight',
        shape=[in_dim, out_dim],
        dtype=tf.float32,
        initializer=initializer_w,
        trainable=True)
    b = tf.get_variable(
        name='bias',
        shape=[out_dim],
        dtype=tf.float32,
        initializer=initializer_b,
        trainable=True)
    return tf.add(tf.matmul(x, W), b)

