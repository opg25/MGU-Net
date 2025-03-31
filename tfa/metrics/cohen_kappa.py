import tensorflow as tf
from tensorflow.keras.metrics import Metric

class CohenKappa(Metric):
    def __init__(self, num_classes, name='cohen_kappa', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.confusion_matrix = self.add_weight(name="confusion_matrix", 
                                              shape=(num_classes, num_classes), 
                                              initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Flatten the predictions and true values
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.argmax(y_pred, axis=-1)
        y_pred = tf.reshape(y_pred, [-1])
        
        # Cast to same dtype
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)

        # Calculate confusion matrix
        current_cm = tf.cast(
            tf.math.confusion_matrix(
                y_true, 
                y_pred,
                num_classes=self.num_classes,
                dtype=tf.int32
            ),
            dtype=tf.float32
        )

        # Update the total confusion matrix
        self.confusion_matrix.assign_add(current_cm)

    def result(self):
        cm = self.confusion_matrix
        n_classes = tf.cast(self.num_classes, tf.float32)
        
        # Calculate observed agreement
        n = tf.reduce_sum(cm)
        sum_po = tf.reduce_sum(tf.linalg.diag_part(cm))
        po = sum_po / n
        
        # Calculate expected agreement
        row_sums = tf.reduce_sum(cm, axis=1)
        col_sums = tf.reduce_sum(cm, axis=0)
        pe = tf.reduce_sum(row_sums * col_sums) / (n * n)
        
        # Calculate kappa
        kappa = (po - pe) / (1 - pe + tf.keras.backend.epsilon())
        return kappa

    def reset_states(self):
        self.confusion_matrix.assign(tf.zeros_like(self.confusion_matrix))
