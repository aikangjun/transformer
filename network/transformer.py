from network import *
from network.encoder import Encoder
from network.decoder import Decoder
import tensorflow as tf


class Transformer(models.Model):
    def __init__(self,
                 num_block: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 embedding_size: int,
                 num_heads: int,
                 dropout: float,
                 **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.num_block = num_block
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.dropout = dropout

        self.encoder = Encoder(num_encoder_block=num_block,
                               vocab_size=src_vocab_size,
                               embedding_size=embedding_size,
                               num_heads=num_heads,
                               dropout=dropout)
        self.decoder = Decoder(num_decoder_block=num_block,
                               vocab_size=tgt_vocab_size,
                               embedding_size=embedding_size,
                               num_heads=num_heads,
                               dropout=dropout)

    def get_config(self):
        # 在Model中get_config()方法未实现
        # 不能使用config = super(Transformer, self).get_config()
        config = {}
        config.update({
            'num_block': self.num_block,
            'src_vocab_size': self.src_vocab_size,
            'tgt_vocab_size': self.tgt_vocab_size,
            'embedding_size': self.embedding_size,
            'num_heads': self.num_heads,
            'dropout': self.dropout
        })
        return config

    def context_padding_mask(self, inputs):
        '''
        上下文掩码
        :param inputs:(batch_size,seq_lens)
        :return:返回形状为(batch-size,seq_lens,seq_lens)
        '''
        # src_seq_lens 和 tgt_seq_len 都是填充到相同维度进行
        context_padding_mask = tf.expand_dims(tf.cast(tf.equal(inputs, 0), tf.float32), axis=-1)
        # 该padding_mask 是与attention形状相同，是由query和key相乘后的mask
        # 需要query_mask和key_mask做同样的操作得到
        context_padding_mask = tf.matmul(context_padding_mask, tf.transpose(context_padding_mask, perm=[0, 2, 1]))
        return context_padding_mask

    def call(self, inputs, training=None, mask=None):
        assert inputs is list or tuple, 'inputs必须是一个list或者tuple,里面包含encoder_inputs,decoder_inputs'
        encoder_inputs, decoder_inputs = inputs

        encoder_outputs, _ = self.encoder(encoder_inputs)
        context_attention_mask = self.context_padding_mask(decoder_inputs)
        decoder_outputs, _, _ = self.decoder(decoder_inputs, encoder_outputs, context_attention_mask)
        return decoder_outputs


if __name__ == '__main__':
    src_seq = tf.constant([[1, 1, 1, 0, 0, 0], [2, 2, 2, 0, 0, 0]])
    tgt_seq = tf.constant([[2, 2, 2, 0, 0, 0], [3, 3, 3, 0, 0, 0]])
    transformer = Transformer(num_block=6,
                              src_vocab_size=10,
                              tgt_vocab_size=20,
                              embedding_size=128,
                              num_heads=8,
                              dropout=0.1)
    outputs = transformer([src_seq,tgt_seq])

