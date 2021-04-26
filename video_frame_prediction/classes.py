from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
import numpy as np
from tensorflow.keras.models import Model

layers=keras.layers.Layer


#-----------------------------------------------------------------------------------------------------------------
class AutoEncoder(Model):
  def __init__(self):
    super(AutoEncoder, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(28, 28, 1)),
      layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
      layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),
      layers.Flatten(),
      layers.Dense(256, activation='relu')])

    self.decoder = tf.keras.Sequential([
      layers.reshape(target_shape=(8, 8, 4)),
      layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
      layers.Conv2D(1, kernel_size=(3, 3), activation='linear', padding='same')])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

  # -----------------------------------------------------------------------------------------------------------------

class MultiAttention(Layer): #Multihead attention
  def __init__(self, d_k, d_v, n_heads,filt_dim):
    super(MultiAttention, self).__init__()
    self.d_k = d_k
    self.d_v = d_v
    self.n_heads = n_heads
    self.filt_dim=filt_dim
    self.attn_heads = list()

  def build(self, input_shape):
    for n in range(self.n_heads):
      self.attn_heads.append(SingleAttention(self.d_k, self.d_v))
    self.linear = Dense(self.filt_dim, input_shape=input_shape, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform')

  def call(self, inputs):
    attn = [self.attn_heads[i](inputs) for i in range(self.n_heads)]
    concat_attn = tf.concat(attn, axis=-1)
    multi_linear = self.linear(concat_attn)
    return multi_linear

#-----------------------------------------------------------------------------------------------------------------

class SingleAttention(Layer):  # Attention layer
  def __init__(self, d_k, d_v):
    super(SingleAttention, self).__init__()
    self.d_k = d_k
    self.d_v = d_v

  def build(self, input_shape):
    self.query = Dense(self.d_k, input_shape=input_shape, kernel_initializer='glorot_uniform',
                       bias_initializer='glorot_uniform')
    self.key = Dense(self.d_k, input_shape=input_shape, kernel_initializer='glorot_uniform',
                     bias_initializer='glorot_uniform')
    self.value = Dense(self.d_v, input_shape=input_shape, kernel_initializer='glorot_uniform',
                       bias_initializer='glorot_uniform')

  def call(self, inputs):  # inputs = (in_seq, in_seq, in_seq)
    q = self.query(inputs[0])
    k = self.key(inputs[1])

    attn_weights = tf.matmul(q, k, transpose_b=True)
    attn_weights = tf.map_fn(lambda x: x / np.sqrt(self.d_k), attn_weights)
    attn_weights = tf.nn.softmax(attn_weights, axis=-1)

    v = self.value(inputs[2])
    attn_out = tf.matmul(attn_weights, v)
    return attn_out

  # -----------------------------------------------------------------------------------------------------------------

class Time2Vector(Layer):  # Time embedding layer
  def __init__(self, seq_len, **kwargs):
    super(Time2Vector, self).__init__()
    self.seq_len = seq_len

  def build(self, input_shape):
    self.weights_linear = self.add_weight(name='weight_linear',
                                          shape=(int(self.seq_len),),
                                          initializer='uniform',
                                          trainable=True)

    self.bias_linear = self.add_weight(name='bias_linear',
                                       shape=(int(self.seq_len),),
                                       initializer='uniform',
                                       trainable=True)

    self.weights_periodic = self.add_weight(name='weight_periodic',
                                            shape=(int(self.seq_len),),
                                            initializer='uniform',
                                            trainable=True)

    self.bias_periodic = self.add_weight(name='bias_periodic',
                                         shape=(int(self.seq_len),),
                                         initializer='uniform',
                                         trainable=True)

  def call(self, x):
    x = tf.math.reduce_mean(x[:, :, :], axis=-1)  # Convert (batch, seq_len, 5) to (batch, seq_len)
    time_linear = self.weights_linear * x + self.bias_linear
    time_linear = tf.expand_dims(time_linear, axis=-1)  # (batch, seq_len, 1)

    time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
    time_periodic = tf.expand_dims(time_periodic, axis=-1)  # (batch, seq_len, 1)
    return tf.concat([time_linear, time_periodic], axis=-1)  # (batch, seq_len, 2)

  # -----------------------------------------------------------------------------------------------------------------

class TransformerEncoder(Layer):  # Combining everything into a Transformer encoder
  def __init__(self, d_k, d_v, n_heads, ff_dim, filt_dim, dropout=0.1, **kwargs):
    super(TransformerEncoder, self).__init__()
    self.d_k = d_k
    self.d_v = d_v
    self.n_heads = n_heads
    self.ff_dim = ff_dim
    self.filt_dim = filt_dim
    self.attn_heads = list()
    self.dropout_rate = dropout

  def build(self, input_shape):
    self.attn_multi = MultiAttention(self.d_k, self.d_v, self.n_heads, self.filt_dim)
    self.attn_dropout = Layer.Dropout(self.dropout_rate)
    self.attn_normalize = Layer.LayerNormalization(input_shape=input_shape, epsilon=1e-6)

    self.ff_conv1D_1 = Layer.Conv1D(filters=self.ff_dim, kernel_size=1, activation='relu')
    self.ff_conv1D_2 = Layer.Conv1D(filters=self.filt_dim,
                                    kernel_size=1)  # input_shape[0]=(batch, seq_len, 7), input_shape[0][-1]=7
    self.ff_dropout = Layer.Dropout(self.dropout_rate)
    self.ff_normalize = Layer.LayerNormalization(input_shape=input_shape, epsilon=1e-6)

  def call(self, inputs):  # inputs = (in_seq, in_seq, in_seq)
    attn_layer = self.attn_multi(inputs)
    attn_layer = self.attn_dropout(attn_layer)
    attn_layer = self.attn_normalize(inputs[0] + attn_layer)

    ff_layer = self.ff_conv1D_1(attn_layer)
    ff_layer = self.ff_conv1D_2(ff_layer)
    ff_layer = self.ff_dropout(ff_layer)
    ff_layer = self.ff_normalize(inputs[0] + ff_layer)
    return ff_layer

    # -----------------------------------------------------------------------------------------------------------------