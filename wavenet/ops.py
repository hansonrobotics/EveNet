from __future__ import division

import tensorflow as tf


def create_adam_optimizer(learning_rate, momentum):
    return tf.train.AdamOptimizer(learning_rate=learning_rate,
                                  epsilon=1e-4)


def create_sgd_optimizer(learning_rate, momentum):
    return tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                      momentum=momentum)


def create_rmsprop_optimizer(learning_rate, momentum):
    return tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                     momentum=momentum,
                                     epsilon=1e-5)


optimizer_factory = {'adam': create_adam_optimizer,
                     'sgd': create_sgd_optimizer,
                     'rmsprop': create_rmsprop_optimizer}


def time_to_batch(value, dilation, name=None):
    with tf.name_scope('time_to_batch'):
        shape = tf.shape(value)
        pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
        padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
        reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])
        return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])


def batch_to_time(value, dilation, name=None):
    with tf.name_scope('batch_to_time'):
        shape = tf.shape(value)
        prepared = tf.reshape(value, [dilation, -1, shape[2]])
        transposed = tf.transpose(prepared, perm=[1, 0, 2])
        return tf.reshape(transposed,
                          [tf.div(shape[0], dilation), -1, shape[2]])


def causal_conv(value, filter_, dilation, name='causal_conv'):
    with tf.name_scope(name):
        filter_width = tf.shape(filter_)[0]
        if dilation > 1:
            transformed = time_to_batch(value, dilation)
            conv = tf.nn.conv1d(transformed, filter_, stride=1,
                                padding='VALID')
            restored = batch_to_time(conv, dilation)
        else:
            restored = tf.nn.conv1d(value, filter_, stride=1, padding='VALID')
        # Remove excess elements at the end.
        out_width = tf.shape(value)[1] - (filter_width - 1) * dilation
        result = tf.slice(restored,
                          [0, 0, 0],
                          [-1, out_width, -1])
        return result

def mu_law_encode(audio, quantization_channels):
    '''Quantizes waveform amplitudes.'''
    with tf.name_scope('encode'):
        mu = tf.to_float(quantization_channels - 1)
        # Perform mu-law companding transformation (ITU-T, 1988).
        # Minimum operation is here to deal with rare large amplitudes caused
        # by resampling.
        safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
        magnitude = tf.log1p(mu * safe_audio_abs) / tf.log1p(mu)
        signal = tf.sign(audio) * magnitude
        # Quantize signal to the specified number of levels.
        return tf.to_int32((signal + 1) / 2 * mu + 0.5)


def mu_law_decode(output, quantization_channels):
    '''Recovers waveform from quantized values.'''
    with tf.name_scope('decode'):
        mu = quantization_channels - 1
        # Map values back to [-1, 1].
        signal = 2 * (tf.to_float(output) / mu) - 1
        # Perform inverse of mu-law transformation.
        magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
        return tf.sign(signal) * magnitude
def encode_data(data,data_dim):
    """Encodees a given data to softmax probablity distribution using the following rule.
        The data distrubtion is assumed from interval [0, 1] inclusive. This method first 
        splits the interval into 10 intervals of the same size([0, 0.1),[0.1,0.2),...,[0.9,1)), and 
        uses the boundaries as class for softmax classifiction. In this way we have 11 classes.
        To convert the given value to softmax probablities, a simple trick is used. Each given number
        can be represented using upper bound and lower bound of inteveral it is found.
        Lets say we have value 'x' in the dataset and it is found between 'l'(lower bound) and 'u'(
        lower bound). We want to design a function that represent x interms of l and u.
            x = p * l + q * u
        and we want to find p and q where , p + q = 1
        More explanation and justification  found at https://docs.google.com/document/d/1z29ABXsvclBBJi7Q8svldnc5gvj2NizKad0i3mHDuds/edit?usp=sharing
    Arguments:
        data {{tf.Tensor}} -- Tensor containing values of each shape key of frames
        data_dim {int} -- Dimenstion of each frame. This crossponds to number of shapekeys for single frame

    
    Returns:
        tf.Tensor -- Tensor which contains probablity distribution encoded using the above method.
    """
    quantization_channels = 11
    data = tf.reshape(data,[-1,data_dim])
    data_10x = data * 10
    ceil_index = tf.ceil(data_10x + tf.keras.backend.epsilon() * 10)
    floor_index = tf.floor(data_10x)

    data_ceil_prob = tf.reshape(data_10x - floor_index,(-1,))
    data_floor_prob = tf.reshape(1 - data_ceil_prob, (-1,))

    ceil_index = tf.reshape(ceil_index,(-1,))
    floor_index = tf.reshape(floor_index,(-1,))

    idx1 = tf.range(tf.shape(data)[0]) 
    idx2 = tf.range(data_dim) 

    ceil_index = tf.cast(ceil_index,tf.int32)
    floor_index = tf.cast(floor_index,tf.int32)
    
    idx1 = tf.reshape(tf.tile(tf.reshape(idx1,[-1,1]),[1, data_dim]),[-1])
    idx2 = tf.tile(idx2,[tf.shape(data)[0]])
    idx_ceil = tf.stack((idx1, idx2,ceil_index),axis=-1)
    idx_floor = tf.stack((idx1, idx2,floor_index),axis=-1)
    
    softmax_distr_ceil =  tf.scatter_nd(idx_ceil,data_ceil_prob,(tf.shape(data)[0], data_dim, quantization_channels))
    softmax_distr_floor =  tf.scatter_nd(idx_floor,data_floor_prob,(tf.shape(data)[0],data_dim, quantization_channels))

    return softmax_distr_ceil + softmax_distr_floor


def decode_data(data):
    """This function decodes data encoded by 'encode_data' function above.
    
    Arguments:
        data {tf.Tensor} -- Encoded data
    
    Returns:
        tf.Tensor -- Decoded data
    """

    cm = tf.range(0, 1.1, 0.1)
    decoded = cm *data
    return tf.reduce_sum(decoded,axis=-1)