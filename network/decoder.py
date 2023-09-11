from network import *
from custom.customlayers import PositionEncoding, DecoderBlock
import tensorflow as tf


class Decoder(layers.Layer):
    def __init__(self,
                 num_decoder_block: int,
                 vocab_size: int,
                 embedding_size: int,
                 num_heads: int,
                 dropout: float,
                 **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.num_decoder_block = num_decoder_block
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.dropout = dropout

        self.embedding = layers.Embedding(input_dim=vocab_size,
                                          output_dim=embedding_size)
        self.positionencoding = PositionEncoding(embedding_size=embedding_size)
        self.decoderblocks = [DecoderBlock(num_heads=num_heads,
                                           embedding_size=embedding_size,
                                           dropout=dropout,
                                           ffn_num_hiddens=embedding_size * 2,
                                           ffn_num_outputs=embedding_size)
                              for i in range(num_decoder_block)]
        self.dense = layers.Dense(units=vocab_size)
        self.softmax = layers.Softmax()

    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({
            'num_decoder_block': self.num_decoder_block,
            'vocab_size': self.vocab_size,
            'embedding_size': self.embedding_size,
            'num_heads': self.num_heads,
            'dropout': self.dropout
        })

    def padding_mask(self, inputs):
        '''
        标准文本掩码
        :param inputs:
        :return:返回形状为(batch-size,seq_lens,seq_lens)
        '''
        # 序列中通过0进行padding,如果等于0进行mask
        padding_mask = tf.expand_dims(tf.cast(tf.equal(inputs, 0), tf.float32), axis=-1)
        # 该padding_mask 是与attention相乘，是由query和key相乘后的mask
        # 需要query_mask和key_mask做同样的操作得到
        padding_mask = tf.matmul(padding_mask, tf.transpose(padding_mask, perm=[0, 2, 1]))
        return padding_mask

    def sequence_mask(self, inputs):
        """
        上三角掩码，用于遮挡当前字符的后续字符
        :param inputs: (batch_size,seq_lens)
        :return: (1,seq_lens,seq_lens)
        """
        # 为1的位置就是mask的位置
        sequence_mask = 1 - tf.linalg.band_part(tf.ones(shape=(inputs.shape[-1],) * 2), num_lower=0, num_upper=-1)
        sequence_mask = tf.expand_dims(sequence_mask, axis=0)
        return sequence_mask

    def call(self, inputs, encoder_output=None, context_mask=None, use_attention_mask: bool = True, *args, **kwargs):
        assert encoder_output is not None, 'encoder_outputs不能为None,应该为list'
        embedding = self.embedding(inputs)
        output = self.positionencoding(embedding)
        padding_mask = self.padding_mask(inputs)
        sequence_mask = self.sequence_mask(inputs)
        if use_attention_mask:
            attention_mask = tf.cast(tf.greater(padding_mask + sequence_mask, 0), tf.float32)
        else:
            attention_mask = None
        self_attentions, context_attentions = [], []
        for i, decoderblock in enumerate(self.decoderblocks):
            output, self_attention, context_attention = decoderblock(output, encoder_output[i],
                                                                     attention_mask, context_mask)
            self_attentions.append(self_attention)
            context_attentions.append(context_attention)
        output = self.dense(output)
        output = self.softmax(output)
        return output, self_attentions, context_attentions


if __name__ == '__main__':
    # inputs = tf.constant([[1, 1, 1, 0, 0, 0], [2, 2, 2, 0, 0, 0]])
    # encoder_output = [tf.random.normal(shape=(2, 6, 256)) for _ in range(6)]
    # decoder = Decoder(num_decoder_block=6,
    #                   vocab_size=9,
    #                   embedding_size=256,
    #                   num_heads=8,
    #                   dropout=0.1,
    #                   tgt_vocab_size=12)
    # outputs, _, _ = decoder(inputs, encoder_output)
    #
    import numpy as np

    x = np.random.randint(0, 5000, size=(4, 150))
    embedding = layers.Embedding(input_dim=5000,
                                 output_dim=64)
    x_ = embedding(x)
    y_ = embedding.embeddings.numpy()[x]

