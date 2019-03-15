import numpy as np
import tensorflow as tf
import os
import shutil

class PG_Base:
    def __init__(self,
                 run_name,
                 input_shape,
                 n_action,
                 learning_rate,
                 save_path,
                 record_io,
                 record,
                 gpu_fraction):

        self.run_name = run_name
        self.input_shape = input_shape
        self.n_action = n_action
        self.learning_rate = learning_rate
        self.record_io = record_io
        self.record = record

        if save_path[-1] != '/':
            self.save_path = save_path + '/'
        else:
            self.save_path = save_path

        with tf.Graph().as_default():
            self._build_network()
            self._build_other()
            if gpu_fraction is None:
                self.Sess = tf.Session()
            else:
                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
                self.Sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
            self.Sess.run(tf.global_variables_initializer())

        if self.record_io is True:
            self.deal_record_file()

        if self.record is True:
            self.Writer = tf.summary.FileWriter(self.save_path + self.run_name + '/tensorboard', self.Sess.graph)

    def deal_record_file(self):
        if os.path.exists('{}{}'.format(self.save_path, self.run_name)):
            print('the run directory [{}{}] already exists!'.format(self.save_path, self.run_name))
            print('0: exist ')
            print('1: restored the session from checkPoint ')
            print('2: start over and overwrite')
            print('3: create a new run')
            mode = int(input('please select the mode:'))

            if mode == 0:
                exit('you select to exist')
            elif mode == 1:
                self.load(self.save_path, self.run_name)
            elif mode == 2:
                shutil.rmtree('{}{}'.format(self.save_path, self.run_name))
            elif mode == 3:
                self.run_name = input('please enter a new run name:')
                self.deal_record_file()
            else:
                raise ValueError('the valid actions are in range [0-3]')

    def _build_network(self):
        """
        #the template of the _build_network
        #----------------------------------------------------------------------------------------------------------
        self.S = tf.placeholder(tf.float32, [None, 160, 160], name='obs')
        self.R = tf.placeholder(tf.float32, [None], name='reward')
        self.A = tf.placeholder(tf.int32, [None], name='old_action')
        self.input_pixels = 1
        for i in self.input_shape:
            self.input_pixels *= i

        def network(x):
            initializer = tf.contrib.layers.xavier_initializer()

            #--------------------------------- change start here ---------------------------------------------------
            Weight = {'fc1': tf.get_variable('w_fc1', [self.input_pixels, self.n_l1], initializer=initializer),
                      'out': tf.get_variable('w_out', [self.n_l1, self.n_action], initializer=initializer)}

            Bias = {'fc1': tf.get_variable('b_fc1', [self.input_pixels, self.n_l1], initializer=initializer),
                    'out': tf.get_variable('b_out', [self.n_action], initializer=initializer)}

            Flattened = tf.reshape(x, shape=[-1, self.input_pixels])
            L1 = tf.nn.relu(tf.matmul(Flattened, Weight['fc1']) + Bias['fc1'])
            output = tf.matmul(L1, Weight['out']) + Bias['out']

            Summary = [tf.summary.histogram('w_fc1', Weight['fc1']),
                       tf.summary.histogram('w_out', Weight['out']),
                       tf.summary.histogram('b_fc1', Bias['fc1']),
                       tf.summary.histogram('b_out', Bias['out'])]

            #--------------------------------- change stop here ---------------------------------------------------
            return output, Summary

        self.Out, self.Summary_weight = network(self.S)

        self.Prob = tf.nn.softmax(self.Out)
        self.Cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.Out, labels=self.A)
        self.Loss = tf.reduce_mean(self.Cross_entropy * self.R)
        self.Train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.Loss)
        """

        raise NotImplementedError("Subclass should implement this")

    def _build_other(self):
        with tf.name_scope('reflect'):
            self.Loss_reflect = tf.placeholder(tf.float32, shape=None)
            self.Reward_reflect = tf.placeholder(tf.float32, shape=None)

        with tf.name_scope('step_counter'):
            self.Step = tf.Variable(tf.constant(0), dtype=tf.int32)
            self.Step_move = tf.assign(self.Step, self.Step + tf.constant(1))

        with tf.name_scope('summary'):
            self.Summary_loss = tf.summary.scalar('loss', self.Loss_reflect)
            self.Summary_reward = tf.summary.scalar('total_reward', self.Reward_reflect)

        self.Saver = tf.train.Saver()

    def choose_action(self, obs):
        obs = np.expand_dims(obs, axis=0)
        prob = self.Sess.run(self.Prob, feed_dict={self.S: obs})[0]
        action = np.random.choice(self.n_action, p=prob)

        return action

    def get_action_prob(self, obs):
        obs = np.expand_dims(obs, axis=0)
        prob = self.Sess.run(self.Prob, feed_dict={self.S: obs})[0]
        return prob

    def random_action(self):
        return np.random.choice(self.n_action)

    def train(self, s, a, r, record):
        _, loss = self.Sess.run([self.Train, self.Loss], feed_dict={self.S: s,
                                                                    self.A: a,
                                                                    self.R: r})
        if record is True:
            if self.record is False:
                raise ValueError('cannot record due to the agent of record is False')
            result1, result2, step = self.Sess.run([self.Summary_loss, self.Summary_weight, self.Step], feed_dict={self.Loss_reflect: loss})
            self.Writer.add_summary(result1, step)
            self.Writer.add_summary(result2, step)

        return loss

    def save(self):
        self.Saver.save(self.Sess, '{}{}/{}.ckpt'.format(self.save_path, self.run_name, self.run_name))

    def load(self, load_path=None, run_name=None):
        if load_path is None:
            load_path = self.save_path
        if run_name is None:
            run_name = self.run_name
        self.Saver.restore(self.Sess, '{}{}/{}.ckpt'.format(load_path, run_name, run_name))

    def step_move(self):
        step = self.Sess.run(self.Step_move)
        return step

    def log_reward(self, reward):
        if self.record is False:
            raise ValueError('cannot record due to the agent of record is False')

        result, step = self.Sess.run([self.Summary_reward, self.Step], feed_dict={self.Reward_reflect: reward})
        self.Writer.add_summary(result, step)

    def close_model(self):
        self.Sess.close()

    def model_pos_check(self):
        '''
        this extrally save the model graph of self.Sess and sess default for redundant declaration checking
        '''

        if self.record is False:
            raise ValueError('cannot do model_pos_check if record is set to False')
        else:
            if not hasattr(self, 'Sess_default'):
                self.Sess_default = tf.Session()

            tf.summary.FileWriter(self.save_path + self.run_name + '/model_check/sess/', self.Sess.graph)
            tf.summary.FileWriter(self.save_path + self.run_name + '/model_check/sess_default', self.Sess_default.graph)
