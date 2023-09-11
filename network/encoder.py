from network import *
from custom.customlayers import PositionEncoding, EncoderBlock
import tensorflow as tf


class Encoder(layers.Layer):
    def __init__(self,
                 num_encoder_block: int,
                 vocab_size: int,
                 embedding_size: int,
                 num_heads: int,
                 dropout: float,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.num_encoder_block = num_encoder_block
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.dropout = dropout

        self.embedding = layers.Embedding(input_dim=vocab_size,
                                          output_dim=embedding_size)
        self.positionencoding = PositionEncoding(embedding_size=embedding_size)
        self.encoderblocks = [EncoderBlock(embedding_size=embedding_size,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          ffn_num_hiddens=embedding_size*2,
                                          ffn_num_outputs=embedding_size) for i in range(num_encoder_block)]

    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            'num_encoder_block': self.num_encoder_block,
            'vocab_size': self.vocab_size,
            'embedding_size': self.embedding_size,
            'num_heads': self.num_heads,
            'dropout': self.dropout
        })
        return config

    def padding_mask(self,inputs):
        '''
        标注文本掩码
        :param inputs:(batch_size,seq_lens)
        :return:返回形状为(batch-size,seq_lens,seq_lens)
        '''
        padding_mask = tf.expand_dims(tf.cast(tf.equal(inputs,0),tf.float32),axis=-1)
        # 该padding_mask 是与attention形状相同，是由query和key相乘后的mask
        # 需要query_mask和key_mask做同样的操作得到
        padding_mask = tf.matmul(padding_mask,tf.transpose(padding_mask,perm=[0,2,1]))
        return padding_mask

    def call(self, inputs, *args, **kwargs):
        embedding = self.embedding(inputs)
        embedding = self.positionencoding(embedding)
        attention_mask = self.padding_mask(inputs)
        encoder_outputs,encoder_attentions = [],[]
        for encoderblock in self.encoderblocks:
            output,attention = encoderblock(embedding,attention_mask)
            encoder_outputs.append(output)
            encoder_attentions.append(attention)
        return encoder_outputs,encoder_attentions
if __name__ == '__main__':
    inputs = tf.constant([[1,2,3,4,5,0,0,0,0],[1,2,3,4,5,0,0,0,0]])
    encoder = Encoder(num_encoder_block=6,
                      vocab_size=7,
                      embedding_size=128,
                      num_heads=8,
                      dropout=0.1)
    outputs,_ = encoder(inputs)
