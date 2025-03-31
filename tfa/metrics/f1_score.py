import tensorflow as tf
from tensorflow.keras.metrics import Metric

class F1Score(Metric):
    def __init__(self, num_classes, average='macro', name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.average = average
        self.true_positives = self.add_weight(name='true_positives', shape=(num_classes,), initializer='zeros')
        self.false_positives = self.add_weight(name='false_positives', shape=(num_classes,), initializer='zeros')
        self.false_negatives = self.add_weight(name='false_negatives', shape=(num_classes,), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1)
        y_pred = tf.cast(y_pred, tf.int32)

        # Convert y_true and y_pred to one-hot encoding
        y_true_oh = tf.one_hot(y_true, self.num_classes)
        y_pred_oh = tf.one_hot(y_pred, self.num_classes)

        # Calculate true positives, false positives, and false negatives per class
        true_pos = tf.reduce_sum(y_true_oh * y_pred_oh, axis=[0, 1])
        false_pos = tf.reduce_sum(y_pred_oh * (1 - y_true_oh), axis=[0, 1])
        false_neg = tf.reduce_sum((1 - y_pred_oh) * y_true_oh, axis=[0, 1])

        # Update states using assign_add
        self.true_positives.assign_add(true_pos)
        self.false_positives.assign_add(false_pos)
        self.false_negatives.assign_add(false_neg)

    def result(self):
        # Calculate precision and recall
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        
        # Calculate F1 score
        f1_score = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

        if self.average == 'macro':
            return tf.reduce_mean(f1_score)
        elif self.average == 'weighted':
            weights = tf.reduce_sum(self.true_positives + self.false_negatives, axis=0)
            return tf.reduce_sum(f1_score * weights) / tf.reduce_sum(weights)
        return f1_score

    def reset_states(self):
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_positives.assign(tf.zeros_like(self.false_positives))
        self.false_negatives.assign(tf.zeros_like(self.false_negatives))
