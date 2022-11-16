import numpy as np
import os
import tensorflow as tf
import logging
log = logging.getLogger(__name__)

def sgd(X, y_true, model, loss, batch_size=4, epochs=1, shuffle=False, lr=1e-3):
    for epoch in range(1, epochs+1):
        #log.info("EPOCH: {}".format(epoch))
        if shuffle:
            indices = np.random.permutation(X.shape[0])
        else:
            indices = np.arange(X.shape[0])
        for i in range(X.shape[0]//batch_size):
            start = i*batch_size
            end = start + batch_size
            X_batch = X[indices[start:end], :]
            y_batch = y_true[indices[start:end], :]
            sgd_step(X_batch.T, y_batch.T, model.next, loss, lr)
    return

def sgd_step(X, y, layer, loss, lr):
    if layer is None:
        log.debug("y_pred {}".format(X))
        log.debug("y_true {}".format(y))
        log.info("loss {}".format(loss.forward(X, y)))
        delta = loss.backward(X, y)
        log.debug("delta {}".format(delta))
        return delta
    else:
        log.debug(layer.__class__.__name__)
        log.debug("input {}".format(X))
        a = layer.forward(X)
        log.debug("output {}".format(a))
        delta = sgd_step(a, y, layer.next, loss, lr)
        log.debug("delta {}".format(delta))
        grad = layer.backward(X, delta)
        log.debug("grad \n{}".format(grad))
        if layer.trainable:
            layer.update(X, delta, lr)
            log.debug("NEW VALUES {}".format(layer.get_weights()))
        return grad




class Network(object):
    """Creates the computation graph for a fully connected rectifier/dropout network
     prediction model and Fisher diagonal."""

    def __init__(self, num_features, num_class, fc_hidden_units, apply_dropout, ewc_batch_size=100, ewc_batches=550):
        self.num_features = num_features
        self.num_class = num_class
        self.fc_units = fc_hidden_units
        self.sizes = [self.num_features] + self.fc_units + [self.num_class]
        self.apply_dropout = apply_dropout
        self.ewc_batch_size = ewc_batch_size
        self.ewc_batches = ewc_batches

        self.x = None
        self.y = None
        self.x_fisher = None
        self.y_fisher = None
        self.keep_prob_input = None
        self.keep_prob_hidden = None

        self.biases = None
        self.weights = None
        self.theta = None
        self.biases_lagged = None
        self.weights_lagged = None
        self.theta_lagged = None

        self.scores = None
        self.fisher_diagonal = None
        self.fisher_minibatch = None

        self.fisher_accumulate_op = None
        self.fisher_full_batch_average_op = None
        self.fisher_zero_op = None
        self.update_theta_op = None

        self.create_graph()

        self.saver = tf.train.Saver(max_to_keep=1000, var_list=self.theta + self.theta_lagged + self.fisher_diagonal)

    def create_graph(self):
        self.create_placeholders()
        self.create_fc_variables()
        self.scores = self.fc_feedforward(self.x, self.biases, self.weights, self.apply_dropout)
        self.create_fisher_diagonal()

    def fc_feedforward(self, h, biases, weights, apply_dropout):
        if apply_dropout:
            h = tf.nn.dropout(h, self.keep_prob_input)
        for (w, b) in list(zip(weights, biases))[:-1]:
            h = self.create_fc_layer(h, w, b)
            if apply_dropout:
                h = tf.nn.dropout(h, self.keep_prob_hidden)
        return self.create_fc_layer(h, weights[-1], biases[-1], apply_relu=False)

    def create_fisher_diagonal(self):
        nll, biases_per_example, weights_per_example = self.unaggregated_nll()
        self.fisher_minibatch = self.fisher_minibatch_sum(nll, biases_per_example, weights_per_example)
        self.create_fisher_ops()

    def unaggregated_nll(self):
        x_examples = tf.unstack(self.x_fisher)
        y_examples = tf.unstack(self.y_fisher)
        biases_per_example = [self.clone_variable_list(self.biases) for _ in range(0, self.ewc_batch_size)]
        weights_per_example = [self.clone_variable_list(self.weights) for _ in range(0, self.ewc_batch_size)]
        nll_list = []
        for (x, y, biases, weights) in zip(x_examples, y_examples, biases_per_example, weights_per_example):
            scores = self.fc_feedforward(tf.reshape(x, [1, self.num_features]), biases, weights, apply_dropout=False)
            nll = - tf.reduce_sum(y * tf.nn.log_softmax(scores))
            nll_list.append(nll)
        nlls = tf.stack(nll_list)
        return tf.reduce_sum(nlls), biases_per_example, weights_per_example

    def fisher_minibatch_sum(self, nll_per_example, biases_per_example, weights_per_example):
        bias_grads_per_example = [tf.gradients(nll_per_example, biases) for biases in biases_per_example]
        weight_grads_per_example = [tf.gradients(nll_per_example, weights) for weights in weights_per_example]
        return self.sum_of_squared_gradients(bias_grads_per_example, weight_grads_per_example)

    def sum_of_squared_gradients(self, bias_grads_per_example, weight_grads_per_example):
        bias_grads2_sum = []
        weight_grads2_sum = []
        for layer in range(0, len(self.fc_units) + 1):
            bias_grad2_sum = tf.add_n([tf.square(example[layer]) for example in bias_grads_per_example])
            weight_grad2_sum = tf.add_n([tf.square(example[layer]) for example in weight_grads_per_example])
            bias_grads2_sum.append(bias_grad2_sum)
            weight_grads2_sum.append(weight_grad2_sum)
        return bias_grads2_sum + weight_grads2_sum

    def create_fisher_ops(self):
        self.fisher_diagonal = self.bias_shaped_variables(name='bias_grads2', c=0.0, trainable=False) + \
                               self.weight_shaped_variables(name='weight_grads2', c=0.0, trainable=False)

        self.fisher_accumulate_op = [tf.assign_add(f1, f2) for f1, f2 in
                                     zip(self.fisher_diagonal, self.fisher_minibatch)]
        scale = 1 / float(self.ewc_batches * self.ewc_batch_size)
        self.fisher_full_batch_average_op = [tf.assign(var, scale * var) for var in self.fisher_diagonal]
        self.fisher_zero_op = [tf.assign(tensor, tf.zeros_like(tensor)) for tensor in self.fisher_diagonal]

    @staticmethod
    def create_fc_layer(input, w, b, apply_relu=True):
        with tf.name_scope('fc_layer'):
            output = tf.matmul(input, w) + b
            if apply_relu:
                output = tf.nn.relu(output)
        return output

    @staticmethod
    def create_variable(shape, name, c=None, sigma=None, trainable=True):
        if sigma:
            initial = tf.truncated_normal(shape, stddev=sigma, name=name)
        else:
            initial = tf.constant(c if c else 0.0, shape=shape, name=name)
        return tf.Variable(initial, trainable=trainable)

    @staticmethod
    def clone_variable_list(variable_list):
        return [tf.identity(var) for var in variable_list]

    def bias_shaped_variables(self, name, c=None, sigma=None, trainable=True):
        return [self.create_variable(shape=[i], name=name + '{}'.format(layer + 1),
                                     c=c, sigma=sigma, trainable=trainable) for layer, i in enumerate(self.sizes[1:])]

    def weight_shaped_variables(self, name, c=None, sigma=None, trainable=True):
        return [self.create_variable([i, j], name=name + '{}'.format(layer + 1),
                                     c=c, sigma=sigma, trainable=trainable)
                for layer, (i, j) in enumerate(zip(self.sizes[:-1], self.sizes[1:]))]

    def create_fc_variables(self):
        with tf.name_scope('fc_variables'):
            self.biases = self.bias_shaped_variables(name='biases_fc', c=0.1, trainable=True)
            self.weights = self.weight_shaped_variables(name='weights_fc', sigma=0.1, trainable=True)
            self.theta = self.biases + self.weights
        with tf.name_scope('fc_variables_lagged'):
            self.biases_lagged = self.bias_shaped_variables(name='biases_fc_lagged', c=0.0, trainable=False)
            self.weights_lagged = self.weight_shaped_variables(name='weights_fc_lagged', c=0.0, trainable=False)
            self.theta_lagged = self.biases_lagged + self.weights_lagged
        self.update_theta_op = [v1.assign(v2) for v1, v2 in zip(self.theta_lagged, self.theta)]

    def create_placeholders(self):
        with tf.name_scope("prediction-inputs"):
            self.x = tf.placeholder(tf.float32, [None, self.num_features], name='x-input')
            self.y = tf.placeholder(tf.float32, [None, self.num_class], name='y-input')
        with tf.name_scope("dropout-probabilities"):
            self.keep_prob_input = tf.placeholder(tf.float32)
            self.keep_prob_hidden = tf.placeholder(tf.float32)
        with tf.name_scope("fisher-inputs"):
            self.x_fisher = tf.placeholder(tf.float32, [self.ewc_batch_size, self.num_features])
            self.y_fisher = tf.placeholder(tf.float32, [self.ewc_batch_size, self.num_class])


class Classifier(Network):
    """Supplies fully connected prediction model with training loop which absorbs minibatches and updates weights."""

    def __init__(self, checkpoint_path='logs/checkpoints/', summaries_path='logs/summaries/', *args, **kwargs):
        super(Classifier, self).__init__(*args, **kwargs)
        self.checkpoint_path = checkpoint_path
        self.summaries_path = summaries_path
        self.writer = None
        self.merged = None
        self.optimizer = None
        self.train_step = None
        self.accuracy = None
        self.loss = None

        self.create_loss_and_accuracy()

    def train(self, sess, model_name, model_init_name, dataset, num_updates, mini_batch_size, fisher_multiplier,
              learning_rate, log_frequency=None, dataset_lagged=None):  # pass previous dataset as convenience
        print('training ' + model_name + ' with weights initialized at ' + str(model_init_name))
        self.prepare_for_training(sess, model_name, model_init_name, fisher_multiplier, learning_rate)
        for i in range(num_updates):
            self.minibatch_sgd(sess, i, dataset, mini_batch_size, log_frequency)
        self.update_fisher_full_batch(sess, dataset)
        self.save_weights(i, sess, model_name)
        print('finished training ' + model_name)

    def test(self, sess, model_name, batch_xs, batch_ys):
        self.restore_model(sess, model_name)
        feed_dict = self.create_feed_dict(batch_xs, batch_ys, keep_input=1.0, keep_hidden=1.0)
        accuracy = sess.run(self.accuracy, feed_dict=feed_dict)
        return accuracy

    def minibatch_sgd(self, sess, i, dataset, mini_batch_size, log_frequency):
        batch_xs, batch_ys = dataset.next_batch(mini_batch_size)
        feed_dict = self.create_feed_dict(batch_xs, batch_ys)
        sess.run(self.train_step, feed_dict=feed_dict)
        if log_frequency and i % log_frequency is 0:
            self.evaluate(sess, i, feed_dict)

    def evaluate(self, sess, iteration, feed_dict):
        if self.apply_dropout:
            feed_dict.update({self.keep_prob_input: 1.0, self.keep_prob_hidden: 1.0})
        summary, accuracy = sess.run([self.merged, self.accuracy], feed_dict=feed_dict)
        self.writer.add_summary(summary, iteration)

    def update_fisher_full_batch(self, sess, dataset):
        dataset._index_in_epoch = 0  # ensures that all training examples are included without repetitions
        sess.run(self.fisher_zero_op)
        for _ in range(0, self.ewc_batches):
            self.accumulate_fisher(sess, dataset)
        sess.run(self.fisher_full_batch_average_op)
        sess.run(self.update_theta_op)

    def accumulate_fisher(self, sess, dataset):
        batch_xs, batch_ys = dataset.next_batch(self.ewc_batch_size)
        sess.run(self.fisher_accumulate_op, feed_dict={self.x_fisher: batch_xs, self.y_fisher: batch_ys})

    def prepare_for_training(self, sess, model_name, model_init_name, fisher_multiplier, learning_rate):
        self.writer = tf.summary.FileWriter(self.summaries_path + model_name, sess.graph)
        self.merged = tf.summary.merge_all()
        self.train_step = self.create_train_step(fisher_multiplier if model_init_name else 0.0, learning_rate)
        init = tf.global_variables_initializer()
        sess.run(init)
        if model_init_name:
            self.restore_model(sess, model_init_name)

    def create_loss_and_accuracy(self):
        with tf.name_scope("loss"):
            average_nll = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.y))  # optimized
            tf.summary.scalar("loss", average_nll)
            self.loss = average_nll
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.scores, 1), tf.argmax(self.y, 1)), tf.float32))
            tf.summary.scalar('accuracy', accuracy)
            self.accuracy = accuracy

    def create_train_step(self, fisher_multiplier, learning_rate):
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            penalty = tf.add_n([tf.reduce_sum(tf.square(w1 - w2) * f) for w1, w2, f
                                in zip(self.theta, self.theta_lagged, self.fisher_diagonal)])
            return self.optimizer.minimize(self.loss + (fisher_multiplier / 2) * penalty, var_list=self.theta)

    def save_weights(self, time_step, sess, model_name):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        self.saver.save(sess=sess, save_path=self.checkpoint_path + model_name + '.ckpt', global_step=time_step,
                        latest_filename=model_name)
        print('saving model ' + model_name + ' at time step ' + str(time_step))

    def restore_model(self, sess, model_name):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir=self.checkpoint_path, latest_filename=model_name)
        self.saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)

    def create_feed_dict(self, batch_xs, batch_ys, keep_hidden=0.5, keep_input=0.8):
        feed_dict = {self.x: batch_xs, self.y: batch_ys}
        if self.apply_dropout:
            feed_dict.update({self.keep_prob_hidden: keep_hidden, self.keep_prob_input: keep_input})
        return feed_dict
