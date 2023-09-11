import numpy as np

from network.transformer import Transformer
import tensorflow as tf
import tensorflow.keras as keras
from _utils.utils import WarmUpCosineDecayScheduler
from _utils.generator import Generater
from custom.customloss import MaskedSparseCategoricalCrossentropy


class TransformerModel():
    def __init__(self,
                 num_block: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 embedding_size: int,
                 num_heads: int,
                 dropout: float,
                 lr: float):
        self.lr = lr

        self.network = Transformer(num_block=num_block,
                                   src_vocab_size=src_vocab_size,
                                   tgt_vocab_size=tgt_vocab_size,
                                   embedding_size=embedding_size,
                                   num_heads=num_heads,
                                   dropout=dropout)
        self.loss_fn = MaskedSparseCategoricalCrossentropy()
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)

        self.train_loss = keras.metrics.Mean()
        self.train_acc = keras.metrics.SparseCategoricalAccuracy()

        self.val_loss = keras.metrics.Mean()
        self.val_acc = keras.metrics.SparseCategoricalAccuracy()

    def train(self, sources, logits, targets):
        with tf.GradientTape() as tape:
            predict = self.network([sources, logits])
            loss = self.loss_fn(targets, predict)
        gridients = tape.gradient(loss, self.network.trainable_variables)
        if np.isnan(loss):
            raise ValueError('loss出现nan')
        elif loss > 10.0:
            raise ValueError('loss大于10')

        self.optimizer.apply_gradients(zip(gridients, self.network.trainable_variables))
        self.train_loss(loss)
        self.train_acc(targets, predict)

    def predict_seq2seq(self, source, num_step, gen: Generater):
        enc_X = tf.expand_dims(source, axis=0)
        enc_outputs, enc_attention = self.network.encoder(enc_X)

        dec_x = tf.expand_dims(tf.constant(gen.token2index(['[START]'])), axis=0)
        output_seq = []
        for _ in range(num_step):
            Y, _, _ = self.network.decoder(dec_x, enc_outputs, use_attention_mask=True)
            Y = tf.squeeze(Y, axis=0)
            Y = tf.argmax(Y, axis=-1)
            pred = Y
            start = dec_x[0][0]
            dec_x = tf.concat([tf.reshape(start, shape=(1, 1)), tf.cast(tf.expand_dims(Y, axis=0), dtype=tf.int32)],
                              axis=-1)
            if pred[-1] == gen.vocab['[END]']:
                output_seq = list(pred)
                break

        return ''.join(gen.index2token(output_seq))
