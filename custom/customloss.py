from custom import *

class MaskedSparseCategoricalCrossentropy(losses.Loss):
    """
    自定义掩码误差
    """
    def __init__(self, **kwargs):
        super(MaskedSparseCategoricalCrossentropy, self).__init__(**kwargs)
        self.base_loss = losses.SparseCategoricalCrossentropy(from_logits=False, reduction=losses.Reduction.NONE)

    def call(self, y_true, y_pred):

        loss = self.base_loss(y_true, y_pred)
        # 排除掩码误差
        masked_y_true = tf.cast(tf.not_equal(y_true, 0), dtype=loss.dtype)

        masked_loss = tf.boolean_mask(loss, masked_y_true)

        return tf.reduce_mean(masked_loss)