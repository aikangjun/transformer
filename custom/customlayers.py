from custom import *
import numpy as np
import tensorflow as tf


class PositionEncoding(layers.Layer):
    def __init__(self,
                 embedding_size: int,
                 max_seq_len: int = 1000,
                 **kwargs):
        super(PositionEncoding, self).__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.embedding_size = embedding_size
        # pos 是指词语在序列中的位置。i 表示在编码的位置
        # 可以看出，在偶数位置，使用正弦编码，在奇数位置，使用余弦编码。
        self.position_encoding = np.zeros(shape=(1, max_seq_len, embedding_size))
        x = np.arange(max_seq_len).reshape(-1, 1) / (10000 ** (np.arange(0, embedding_size, 2) / embedding_size))
        self.position_encoding[:, :, 0::2] = tf.sin(x)
        self.position_encoding[:, :, 1::2] = tf.cos(x)

    def call(self, inputs, *args, **kwargs):
        outputs = inputs + self.position_encoding[:, :inputs.shape[1], :]
        return outputs


class MultiHeadAttention(layers.Layer):
    '''
    多头注意力机制层
    '''

    def __init__(self,
                 num_heads: int,
                 embedding_size: int,
                 dropout: float,
                 kernel_initializer=initializers.GlorotUniform(),
                 bias_initializer=initializers.Zeros(),
                 kernel_regularizer=regularizers.l2(),
                 bias_regularizer=regularizers.l2(),
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_attention_bias=False,
                 **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.embedding_size = embedding_size
        if dropout:
            self.dropout = dropout
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.use_attention_bias = use_attention_bias

    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'embedding_size': self.embedding_size,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'bias_constraint': self.bias_constraint,
            'use_attention_bias': self.use_attention_bias,
            'use_attention_activation': self.use_attention_activation,
            'dropout': self.dropout
        })
        return config

    def build(self, input_shape):  # input_shape 是一个 TensorShape 类型对象，提供输入的形状
        '''
        在第一次使用该层的时候调用该部分代码，在这里创建变量可以使得变量的形状自适应输入的形状而不需要使用者额外指定变量形状。
        如果已经可以完全确定变量的形状，也可以在__init__部分创建变量
        '''
        # input_shape 为一个list,包含query_shape,key_shape,value_shape
        self.w_q = self.add_weight(name='w_q',
                                   shape=(input_shape[0][-1], self.embedding_size),
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint,
                                   trainable=True)
        self.w_k = self.add_weight(name='w_k',
                                   shape=(input_shape[1][-1], self.embedding_size),
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint,
                                   trainable=True)
        self.w_v = self.add_weight(name='w_v',
                                   shape=(input_shape[-1][-1], self.embedding_size),
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint,
                                   trainable=True)
        self.w_o = self.add_weight(name='w_o',
                                   shape=(self.embedding_size, self.embedding_size),
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint,
                                   trainable=True)
        if self.use_attention_bias:
            self.attention_bias = self.add_weight(name='attention_bias',
                                                  shape=(1,),
                                                  initializer=self.bias_initializer,
                                                  regularizer=self.bias_regularizer,
                                                  constraint=self.bias_constraint,
                                                  trainable=True)
        if hasattr(self, 'dropout'):
            self.drop = layers.Dropout(rate=self.dropout)
        self.built = True

    def call(self, inputs: list or tuple, mask=None, *args, **kwargs):
        assert inputs.__len__() == 3, '多头注意力层需要输入的inputs应该包含三个值'
        # inputs为一个list或者tuple 包含[query,key,value]三个值
        # 多头注意力机制只做一次乘法 得到q,k,v的形状[batch_size,seq_len,embedding_size*num_heads]
        q = tf.matmul(inputs[0], self.w_q)
        k = tf.matmul(inputs[1], self.w_k)
        v = tf.matmul(inputs[-1], self.w_v)
        # 将[batch_size,seq_len,embedding_size*num_heads]->
        # [batch_size,seq_len,num_heads,embedding_size] ->
        # [batch_size,num_heads,seq_len,embedding_size] ->
        # [batch_size*num_heads,seq_len,embedding_size]
        q = tf.reshape(q, shape=(q.shape[0], q.shape[1], self.num_heads, self.embedding_size // self.num_heads))
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        q = tf.reshape(q, shape=(-1, q.shape[2], q.shape[3]))
        k = tf.reshape(k, shape=(k.shape[0], k.shape[1], self.num_heads, self.embedding_size // self.num_heads))
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        k = tf.reshape(k, shape=(-1, k.shape[2], k.shape[3]))
        v = tf.reshape(v, shape=(v.shape[0], v.shape[1], self.num_heads, self.embedding_size // self.num_heads))
        v = tf.transpose(v, perm=[0, 2, 1, 3])
        v = tf.reshape(v, shape=(-1, v.shape[2], v.shape[3]))

        # scaled dot_product attention
        # (batch_size*num_heads,seq_lens,seq_lens)
        attention = tf.matmul(k, tf.transpose(q, perm=[0, 2, 1])) / tf.sqrt(
            tf.cast(self.embedding_size // self.num_heads, dtype=tf.float32))
        if self.use_attention_bias:
            attention += self.attention_bias
        if mask is not None:
            mask = tf.repeat(mask, repeats=self.num_heads, axis=0)
            attention -= 1e+9 * mask
        attention = tf.nn.softmax(attention)

        # b为[batch_size*num_heads,seq_len,embedding_size]
        b = tf.transpose(tf.matmul(v, attention, transpose_a=True), perm=[0, 2, 1])
        # b为[batch_size,num_heads,seq_len,embedding_size]
        b = tf.reshape(b, shape=(-1, self.num_heads, b.shape[1], b.shape[2]))
        # [batch_size,seq_len,num_heads,embedding_size]
        b = tf.transpose(b, perm=[0, 2, 1, 3])
        # [batch_size,seq_len,num_heads*embedding_size]
        b = tf.reshape(b, shape=(b.shape[0], b.shape[1], -1))
        b = tf.matmul(b, self.w_o)
        if hasattr(self, 'drop'):
            b = self.drop(b)
        return b, tf.reshape(attention, shape=(-1, self.num_heads, attention.shape[1], attention.shape[2]))


class PositionWiseFeedForward(layers.Layer):
    '''
    基于位置的前馈网络
    '''

    def __init__(self,
                 ffn_num_hiddens: int,
                 ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFeedForward, self).__init__(**kwargs)
        self.dense1 = layers.Dense(units=ffn_num_hiddens)
        self.relu = layers.ReLU()
        self.dense2 = layers.Dense(units=ffn_num_outputs)

    def call(self, inputs, *args, **kwargs):
        x = self.dense1(inputs)
        x = self.relu(x)
        x = self.dense2(x)
        return x


class EncoderBlock(layers.Layer):
    def __init__(self,
                 num_heads: int,
                 embedding_size: int,
                 dropout: float,
                 ffn_num_hiddens: int,
                 ffn_num_outputs: int,
                 **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        # ffn_num_outputs 与 embedding_size相等
        self.multiheadattention = MultiHeadAttention(num_heads=num_heads,
                                                     embedding_size=embedding_size,
                                                     dropout=dropout)
        self.ln1 = layers.LayerNormalization()
        self.ffn = PositionWiseFeedForward(ffn_num_hiddens=ffn_num_hiddens,
                                           ffn_num_outputs=ffn_num_outputs)
        self.ln2 = layers.LayerNormalization()

    def call(self, inputs, mask=None, *args, **kwargs):
        x, attention = self.multiheadattention([inputs] * 3, mask)
        x = tf.add(inputs, x)
        x = self.ln1(x)
        x_hat = tf.reshape(x, shape=(-1, x.shape[-1]))
        x_hat = self.ffn(x_hat)
        x_hat = tf.reshape(x_hat, shape=tf.shape(inputs))
        x = tf.add(x, x_hat)
        x = self.ln2(x)

        return x, attention


class DecoderBlock(layers.Layer):
    def __init__(self,
                 num_heads: int,
                 embedding_size: int,
                 dropout: float,
                 ffn_num_hiddens: int,
                 ffn_num_outputs: int,
                 **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.ffn_num_hiddens = ffn_num_hiddens
        self.ffn_num_outputs = ffn_num_outputs

        self.multiheadattention_1 = MultiHeadAttention(num_heads=num_heads,
                                                       embedding_size=embedding_size,
                                                       dropout=dropout)
        self.ln_1 = layers.LayerNormalization()
        self.multiheadattention_2 = MultiHeadAttention(num_heads=num_heads,
                                                       embedding_size=embedding_size,
                                                       dropout=dropout)
        self.ln_2 = layers.LayerNormalization()
        self.ffn = PositionWiseFeedForward(ffn_num_hiddens=ffn_num_hiddens,
                                           ffn_num_outputs=ffn_num_outputs)
        self.ln_3 = layers.LayerNormalization()

    def get_config(self):
        config = super(DecoderBlock, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'embedding_szie': self.embedding_size,
            'dropout': self.dropout,
            'ffn_num_hiddens': self.ffn_num_hiddens,
            'ffn_num_outputs': self.ffn_num_outputs
        })
        return config

    def call(self, decoder_inputs, encoder_outputs=None, self_mask=None, context_mask=None, *args, **kwargs):
        decoder_outputs, self_attention = self.multiheadattention_1([decoder_inputs, decoder_inputs, decoder_inputs],
                                                                    self_mask)
        decoder_outputs = tf.add(decoder_inputs, decoder_outputs)
        decoder_outputs = self.ln_1(decoder_outputs)

        decoder_outputs_hat, context_attention = self.multiheadattention_2(
            [decoder_outputs, encoder_outputs, encoder_outputs],
            context_mask)
        decoder_outputs = tf.add(decoder_outputs, decoder_outputs_hat)
        decoder_outputs = self.ln_2(decoder_outputs)

        decoder_outputs_hat = self.ffn(decoder_outputs)
        decoder_outputs = tf.add(decoder_outputs_hat, decoder_outputs)
        return decoder_outputs, self_attention, context_attention


if __name__ == '__main__':
    # decoder_inputs = tf.random.normal(shape=(4, 32, 128))
    # encoder_outputs = tf.random.normal(shape=(4, 32, 128))
    # decoder = DecoderBlock(num_heads=8,
    #                        embedding_size=128,
    #                        dropout=0.1,
    #                        ffn_num_outputs=128,
    #                        ffn_num_hiddens=256)
    # decoder_outputs, _, _ = decoder(decoder_inputs, encoder_outputs)
    encoder_inputs = tf.random.normal(shape=(4, 32, 512))
    encoder = EncoderBlock(num_heads=8,
                           embedding_size=512,
                           dropout=0.1,
                           ffn_num_hiddens=512,
                           ffn_num_outputs=512)
    output = encoder(encoder_inputs)
