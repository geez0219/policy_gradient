import tensorflow as tf
from PG_base import PG_Base


class PG(PG_Base):
    def __init__(self,
                 run_name,
                 input_shape,
                 n_action,
                 learning_rate,
                 save_path,
                 record_io,
                 record,
                 gpu_fraction):

        self.n_l1 = 256
        super().__init__(run_name=run_name,
                         input_shape=input_shape,
                         n_action=n_action,
                         learning_rate=learning_rate,
                         save_path=save_path,
                         record_io=record_io,
                         record=record,
                         gpu_fraction=gpu_fraction)

    def _build_network(self):
        self.S = tf.placeholder(tf.float32, [None, 160, 160], name='obs')
        self.R = tf.placeholder(tf.float32, [None], name='reward')
        self.A = tf.placeholder(tf.int32, [None], name='old_action')
        self.input_pixels = 1
        for i in self.input_shape:
            self.input_pixels *= i

        def network(x):
            initializer = tf.contrib.layers.xavier_initializer()
            Weight = {'fc1': tf.get_variable('w_fc1', [self.input_pixels, self.n_l1], initializer=initializer),
                      'out': tf.get_variable('w_out', [self.n_l1, self.n_action], initializer=initializer)}

            Bias = {'fc1': tf.get_variable('b_fc1', [self.n_l1], initializer=initializer),
                    'out': tf.get_variable('b_out', [self.n_action], initializer=initializer)}

            Flattened = tf.reshape(x, shape=[-1, self.input_pixels])
            L1 = tf.nn.relu(tf.matmul(Flattened, Weight['fc1']) + Bias['fc1'])
            output = tf.matmul(L1, Weight['out']) + Bias['out']

            Summary = [tf.summary.histogram('w_fc1', Weight['fc1']),
                       tf.summary.histogram('w_out', Weight['out']),
                       tf.summary.histogram('b_fc1', Bias['fc1']),
                       tf.summary.histogram('b_out', Bias['out'])]

            return output, Summary

        def network2(x):
            Weight = {'fc1': tf.Variable(tf.truncated_normal([self.input_pixels, self.n_l1], stddev=5e-2)),
                      'out': tf.Variable(tf.truncated_normal([self.n_l1, self.n_action], stddev=5e-2))}

            Flattened = tf.reshape(x, shape=[-1, self.input_pixels])
            L1 = tf.nn.relu(tf.matmul(Flattened, Weight['fc1']))
            output = tf.matmul(L1, Weight['out'])

            Summary = [tf.summary.histogram('w_fc1', Weight['fc1']),
                       tf.summary.histogram('w_out', Weight['out'])]

            return output, Summary

        self.Out, Summary = network(self.S)
        self.Summary_weight = tf.summary.merge(Summary)
        self.Prob = tf.nn.softmax(self.Out)
        self.Cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.Out, labels=self.A)
        self.Loss = tf.reduce_mean(self.Cross_entropy * self.R)
        self.Train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.Loss)


