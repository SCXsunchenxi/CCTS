import tensorflow as tf
import functools
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.framework import ops
from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
#from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.distribute import values as ds_values
from tensorflow.python.eager import context
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.util import nest
from tensorflow.python.platform import tf_logging as logging

import numpy as np
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from IPython import display

class TLSTM(object):


# *************************************** Model: T-LSTM ***************************************

    def init_weights(self, input_dim, output_dim, name, std=0.1, reg=None):
        return tf.get_variable(name, shape=[input_dim, output_dim], initializer=tf.random_normal_initializer(0.0, std),
                               regularizer=reg)

    def init_bias(self, output_dim, name):
        return tf.get_variable(name, shape=[output_dim], initializer=tf.constant_initializer(1.0))

    def no_init_weights(self, input_dim, output_dim, name):
        return tf.get_variable(name, shape=[input_dim, output_dim])

    def no_init_bias(self, output_dim, name):
        return tf.get_variable(name, shape=[output_dim])

    def __init__(self, input_dim, output_dim, hidden_dim, fc_dim, train):
        tf.reset_default_graph()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input = tf.placeholder('float', shape=[None, None, self.input_dim])
        self.labels = tf.placeholder('float', shape=[None, output_dim])
        self.time = tf.placeholder('float', shape=[None, None])
        self.keep_prob = tf.placeholder(tf.float32)

        if train == 1:

            # input gate
            self.Wi = self.init_weights(self.input_dim, self.hidden_dim, name='Input_Hidden_weight', reg=None)
            self.Ui = self.init_weights(self.hidden_dim, self.hidden_dim, name='Input_State_weight', reg=None)
            self.bi = self.init_bias(self.hidden_dim, name='Input_Hidden_bias')

            # forget gate
            self.Wf = self.init_weights(self.input_dim, self.hidden_dim, name='Forget_Hidden_weight', reg=None)
            self.Uf = self.init_weights(self.hidden_dim, self.hidden_dim, name='Forget_State_weight', reg=None)
            self.bf = self.init_bias(self.hidden_dim, name='Forget_Hidden_bias')

            # output gate
            self.Wog = self.init_weights(self.input_dim, self.hidden_dim, name='Output_Hidden_weight', reg=None)
            self.Uog = self.init_weights(self.hidden_dim, self.hidden_dim, name='Output_State_weight', reg=None)
            self.bog = self.init_bias(self.hidden_dim, name='Output_Hidden_bias')

            # state
            self.Wc = self.init_weights(self.input_dim, self.hidden_dim, name='Cell_Hidden_weight', reg=None)
            self.Uc = self.init_weights(self.hidden_dim, self.hidden_dim, name='Cell_State_weight', reg=None)
            self.bc = self.init_bias(self.hidden_dim, name='Cell_Hidden_bias')

            # ct-1 decomp
            self.W_decomp = self.init_weights(self.hidden_dim, self.hidden_dim, name='Decomposition_Hidden_weight',
                                              reg=None)
            self.b_decomp = self.init_bias(self.hidden_dim, name='Decomposition_Hidden_bias_enc')

            # fc
            self.Wo = self.init_weights(self.hidden_dim, fc_dim, name='Fc_Layer_weight',
                                        reg=None)  # tf.contrib.layers.l2_regularizer(scale=0.001)
            self.bo = self.init_bias(fc_dim, name='Fc_Layer_bias')

            # fc
            self.W_softmax = self.init_weights(fc_dim, output_dim, name='Output_Layer_weight',
                                               reg=None)  # tf.contrib.layers.l2_regularizer(scale=0.001)
            self.b_softmax = self.init_bias(output_dim, name='Output_Layer_bias')

            # parameter list
            self.var_list = [self.Wi, self.Ui, self.bi, self.Wf, self.Uf, self.bf, self.Wog, self.Uog, self.bog, self.Wc, self.Uc, self.bc, self.W_decomp, self.b_decomp, self.Wo, self.bo, self.W_softmax, self.b_softmax]


        #test
        else:

            self.Wi = self.no_init_weights(self.input_dim, self.hidden_dim, name='Input_Hidden_weight')
            self.Ui = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Input_State_weight')
            self.bi = self.no_init_bias(self.hidden_dim, name='Input_Hidden_bias')

            self.Wf = self.no_init_weights(self.input_dim, self.hidden_dim, name='Forget_Hidden_weight')
            self.Uf = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Forget_State_weight')
            self.bf = self.no_init_bias(self.hidden_dim, name='Forget_Hidden_bias')

            self.Wog = self.no_init_weights(self.input_dim, self.hidden_dim, name='Output_Hidden_weight')
            self.Uog = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Output_State_weight')
            self.bog = self.no_init_bias(self.hidden_dim, name='Output_Hidden_bias')

            self.Wc = self.no_init_weights(self.input_dim, self.hidden_dim, name='Cell_Hidden_weight')
            self.Uc = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Cell_State_weight')
            self.bc = self.no_init_bias(self.hidden_dim, name='Cell_Hidden_bias')

            self.W_decomp = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Decomposition_Hidden_weight')
            self.b_decomp = self.no_init_bias(self.hidden_dim, name='Decomposition_Hidden_bias_enc')

            self.Wo = self.no_init_weights(self.hidden_dim, fc_dim, name='Fc_Layer_weight')
            self.bo = self.no_init_bias(fc_dim, name='Fc_Layer_bias')

            self.W_softmax = self.no_init_weights(fc_dim, output_dim, name='Output_Layer_weight')
            self.b_softmax = self.no_init_bias(output_dim, name='Output_Layer_bias')

            self.var_list = [self.Wi, self.Ui, self.bi, self.Wf, self.Uf, self.bf, self.Wog, self.Uog, self.bog,
                             self.Wc, self.Uc, self.bc, self.W_decomp, self.b_decomp, self.Wo, self.bo, self.W_softmax,
                             self.b_softmax]

        # vanilla single-task loss
        self.cross_entropy = self.get_cost_acc()[0]
        self.set_vanilla_loss()



    def TLSTM_Unit(self, prev_hidden_memory, concat_input):

        # h c x t
        prev_hidden_state, prev_cell = tf.unstack(prev_hidden_memory)
        batch_size = tf.shape(concat_input)[0]
        x = tf.slice(concat_input, [0, 1], [batch_size, self.input_dim])
        t = tf.slice(concat_input, [0, 0], [batch_size, 1])

        # time decay
        T = self.map_elapse_time(t)

        # Decompose the previous cell if there is a elapse time
        C_ST = tf.nn.tanh(tf.matmul(prev_cell, self.W_decomp) + self.b_decomp)
        C_ST_dis = tf.multiply(T, C_ST)
        # if T is 0, then the weight is one
        prev_cell = prev_cell - C_ST + C_ST_dis

        # Input gate
        i = tf.sigmoid(tf.matmul(x, self.Wi) + tf.matmul(prev_hidden_state, self.Ui) + self.bi)
        # Forget Gate
        f = tf.sigmoid(tf.matmul(x, self.Wf) + tf.matmul(prev_hidden_state, self.Uf) + self.bf)
        # Output Gate
        o = tf.sigmoid(tf.matmul(x, self.Wog) + tf.matmul(prev_hidden_state, self.Uog) + self.bog)
        # Candidate Memory Cell
        C = tf.nn.tanh(tf.matmul(x, self.Wc) + tf.matmul(prev_hidden_state, self.Uc) + self.bc)
        # Current Memory cell
        Ct = f * prev_cell + i * C
        # Current Hidden state ht
        current_hidden_state = o * tf.nn.tanh(Ct)

        # c+h
        return tf.stack([current_hidden_state, Ct])

    # Returns all hidden states for the samples in a batch
    def get_states(self):

        batch_size = tf.shape(self.input)[0]

        scan_input_ = tf.transpose(self.input, perm=[2, 0, 1])
        scan_input = tf.transpose(scan_input_)  # scan input is [seq_length x batch_size x input_dim]
        scan_time = tf.transpose(self.time)  # scan_time [seq_length x batch_size]
        # h
        initial_hidden = tf.zeros([batch_size, self.hidden_dim], tf.float32)
        # h+c
        ini_state_cell = tf.stack([initial_hidden, initial_hidden])

        # make scan_time [seq_length x batch_size x 1]
        scan_time = tf.reshape(scan_time, [tf.shape(scan_time)[0], tf.shape(scan_time)[1], 1])
        concat_input = tf.concat([scan_time, scan_input], 2)  # [seq_length x batch_size x input_dim+1]
        packed_hidden_states = tf.scan(self.TLSTM_Unit, concat_input, initializer=ini_state_cell, name='states')
        # h
        all_states = packed_hidden_states[:, 0, :, :]
        return all_states

    # output fc
    def get_output(self, state):
        output = tf.nn.relu(tf.matmul(state, self.Wo) + self.bo)
        output = tf.nn.dropout(output, self.keep_prob)
        output = tf.matmul(output, self.W_softmax) + self.b_softmax
        return output

    # batch output
    def get_outputs(self):  # Returns all the outputs
        # h of a batch
        all_states = self.get_states()
        all_outputs = tf.map_fn(self.get_output, all_states)
        output = tf.reverse(all_outputs, [0])[0, :, :]
        return output

    # time decay
    def map_elapse_time(self, t):

        c1 = tf.constant(1, dtype=tf.float32)
        c2 = tf.constant(2.7183, dtype=tf.float32)

        # T = tf.multiply(self.wt, t) + self.bt
        T = tf.div(c1, tf.log(t + c2), name='Log_elapse_time')
        Ones = tf.ones([1, self.hidden_dim], dtype=tf.float32)

        # time decay matrix
        T = tf.matmul(T, Ones)

        return T

# ************************************* Loss: original L **************************************

    def get_cost_acc(self):
        logits = self.get_outputs()
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=logits))
        y_pred = tf.argmax(logits, 1)
        y = tf.argmax(self.labels, 1)
        return cross_entropy, y_pred, y, logits, self.labels


# ***************************************** Loss: LM ******************************************

    def compute_fisher(self, tsset, sess, num_samples=200, plot_diffs=False, disp_freq=10):
        # computer Fisher information for each parameter

        # initialize Fisher information for most recent task
        self.F_accum = []
        for v in range(len(self.var_list)):
            self.F_accum.append(np.zeros(self.var_list[v].get_shape().as_list()))

        # sampling a random class from softmax
        # probs = tf.nn.softmax(self.y)
        logits = self.get_outputs()
        probs = tf.nn.softmax(logits)

        class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])

        if(plot_diffs):
            # track differences in mean Fisher info
            F_prev = deepcopy(self.F_accum)
            mean_diffs = np.zeros(0)

        fish_gra = tf.gradients(tf.log(probs[0,class_ind]), self.var_list)
        for i in range(num_samples):
            # select random input image
            im_ind = np.random.randint(tsset.shape[0])
            # compute first-order derivatives
            ders = sess.run(fish_gra, feed_dict={self.input: tsset[im_ind:im_ind+1]})
            # square the derivatives and add to total
            for v in range(len(self.F_accum)):
                self.F_accum[v] += np.square(ders[v])
            if(plot_diffs):
                if i % disp_freq == 0 and i > 0:
                    # recording mean diffs of F
                    F_diff = 0
                    for v in range(len(self.F_accum)):
                        F_diff += np.sum(np.absolute(self.F_accum[v]/(i+1) - F_prev[v]))
                    mean_diff = np.mean(F_diff)
                    mean_diffs = np.append(mean_diffs, mean_diff)
                    for v in range(len(self.F_accum)):
                        F_prev[v] = self.F_accum[v]/(i+1)
                    plt.plot(range(disp_freq+1, i+2, disp_freq), mean_diffs)
                    plt.xlabel("Number of samples")
                    plt.ylabel("Mean absolute Fisher difference")
                    display.display(plt.gcf())
                    display.clear_output(wait=True)

        # divide totals by number of samples
        for v in range(len(self.F_accum)):
            self.F_accum[v] /= num_samples

    def star(self):
        # used for saving optimal weights after most recent task training
        self.star_vars = []

        for v in range(len(self.var_list)):
            self.star_vars.append(self.var_list[v].eval())

    def restore(self, sess):
        # reassign optimal weights for latest task
        if hasattr(self, "star_vars"):
            for v in range(len(self.var_list)):
                sess.run(self.var_list[v].assign(self.star_vars[v]))

    def set_vanilla_loss(self):
        self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.cross_entropy)

    def update_lm_loss(self, lam):
        # LM loss
        # lam is weighting for previous task(s) constraints

        if not hasattr(self, "ewc_loss"):
            self.lm_loss = self.cross_entropy

        for v in range(len(self.var_list)):
            self.lm_loss += (lam/2) * tf.reduce_sum(tf.multiply(self.F_accum[v].astype(np.float32),tf.square(self.var_list[v] - self.star_vars[v])))

        self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.lm_loss)

        pm = PM_AdaSFW(ConstrainedOptimizer)
        pm.apply_gradients(self.train_step)




# ************************************ Gradient update: PM ************************************

def _filter_grads(grads_vars_and_constraints):
    # Filter out iterable with grad equal to None
    grads_vars_and_constraints = tuple(grads_vars_and_constraints)
    if not grads_vars_and_constraints:
        return grads_vars_and_constraints
    filtered = []
    vars_with_empty_grads = []
    for gvc in grads_vars_and_constraints:
        grad = gvc[0]
        var = gvc[1]
        if grad is None:
            vars_with_empty_grads.append(var)
        else:
            filtered.append(gvc)
    filtered = tuple(filtered)
    if not filtered:
        raise ValueError("No gradients provided for any variable: %s." %
                         ([v.name for _, v in grads_vars_and_constraints],))
    if vars_with_empty_grads:
        logging.warning(
            "Gradients do not exist for variables %s when minimizing the loss.",
            ([v.name for v in vars_with_empty_grads]))
    return filtered

class ConstrainedOptimizer(tf.keras.optimizers.Optimizer):
    # Base class for Constrained Optimizers

    def __init__(self, name='ConstrainedOptimizer', **kwargs):
        super().__init__(name, **kwargs)

    def set_learning_rate(self, learning_rate):
        self._set_hyper("learning_rate", learning_rate)

    def _aggregate_gradients(self, grads_vars_and_constraints):
        """Returns all-reduced gradients.
                   Args:
                       grads_vars_and_constraints: List of (gradient, variable, constraint) pairs.
                   Returns:
                       A list of all-reduced gradients.
                   """
        grads_and_vars = [(g, v) for g, v, _ in grads_vars_and_constraints]
        filtered_grads_and_vars = _filter_grads(grads_and_vars)

        def all_reduce_fn(distribution, grads_and_vars):
            return distribution.extended.batch_reduce_to(
                ds_reduce_util.ReduceOp.SUM, grads_and_vars)

        if filtered_grads_and_vars:
            reduced = distribute_ctx.get_replica_context().merge_call(
                all_reduce_fn, args=(filtered_grads_and_vars,))
        else:
            reduced = []
        reduced_with_nones = []
        reduced_pos = 0
        for g, _ in grads_and_vars:
            if g is None:
                reduced_with_nones.append(None)
            else:
                reduced_with_nones.append(reduced[reduced_pos])
                reduced_pos += 1
        assert reduced_pos == len(reduced), "Failed to add all gradients"
        return reduced_with_nones

    def _distributed_apply(self, distribution, grads_vars_and_constraints, name, apply_state):

        def apply_grad_to_update_var(var, grad, constraint):
            # Apply gradient to variable
            if isinstance(var, ops.Tensor):
                raise NotImplementedError("Trying to update a Tensor ", var)

            apply_kwargs = {}
            if isinstance(grad, ops.IndexedSlices):
                if var.constraint is not None:
                    raise RuntimeError(
                        "Cannot use a constraint function on a sparse variable.")
                if "apply_state" in self._sparse_apply_args:
                    apply_kwargs["apply_state"] = apply_state
                return self._resource_apply_sparse_duplicate_indices(
                    grad.values, var, grad.indices, **apply_kwargs)

            if "apply_state" in self._dense_apply_args:
                apply_kwargs["apply_state"] = apply_state
            return self._resource_apply_dense(grad, var, constraint, **apply_kwargs)

        eagerly_outside_functions = ops.executing_eagerly_outside_functions()
        update_ops = []
        with ops.name_scope(name or self._name, skip_on_eager=True):
            for grad, var, constraint in grads_vars_and_constraints:
                def _assume_mirrored(grad):
                    if isinstance(grad, ds_values.PerReplica):
                        return ds_values.Mirrored(grad.values)
                    return grad

                grad = nest.map_structure(_assume_mirrored, grad)
                with distribution.extended.colocate_vars_with(var):
                    with ops.name_scope("update" if eagerly_outside_functions else
                                        "update_" + var.op.name, skip_on_eager=True):
                        update_ops.extend(distribution.extended.update(
                            var, apply_grad_to_update_var, args=(grad, constraint), group=False))

            any_symbolic = any(isinstance(i, ops.Operation) or
                               tf_utils.is_symbolic_tensor(i) for i in update_ops)
            if not context.executing_eagerly() or any_symbolic:
                with ops._get_graph_from_inputs(update_ops).as_default():
                    with ops.control_dependencies(update_ops):
                        return self._iterations.assign_add(1, read_value=False)

            return self._iterations.assign_add(1)

    def apply_gradients(self, grads_vars_and_constraints, name=None, experimental_aggregate_gradients=True):
        grads_vars_and_constraints = _filter_grads(grads_vars_and_constraints)
        var_list = [v for (_, v, _) in grads_vars_and_constraints]
        constraint_list = [c for (_, _, c) in grads_vars_and_constraints]

        with backend.name_scope(self._name):
            with ops.init_scope():
                self._create_all_weights(var_list)

            if not grads_vars_and_constraints:
                return control_flow_ops.no_op()

            if distribute_ctx.in_cross_replica_context():
                raise RuntimeError(
                    "`apply_gradients() cannot be called in cross-replica context. "
                    "Use `tf.distribute.Strategy.run` to enter replica "
                    "context.")

            strategy = distribute_ctx.get_strategy()
            # if (not experimental_aggregate_gradients and strategy and isinstance(
            #         strategy.extended,
            #         parameter_server_strategy.ParameterServerStrategyExtended)):
            #     raise NotImplementedError(
            #         "`experimental_aggregate_gradients=False is not supported for "
            #         "ParameterServerStrategy and CentralStorageStrategy")

            apply_state = self._prepare(var_list)
            if experimental_aggregate_gradients:
                reduced_grads = self._aggregate_gradients(grads_vars_and_constraints)
                var_list = [v for _, v, _ in grads_vars_and_constraints]
                grads_vars_and_constraints = list(zip(reduced_grads, var_list, constraint_list))
            return distribute_ctx.get_replica_context().merge_call(
                functools.partial(self._distributed_apply, apply_state=apply_state),
                args=(grads_vars_and_constraints,),
                kwargs={
                    "name": name,
                })

class PM_AdaSFW(ConstrainedOptimizer):
    """
    Arguments:
        learning_rate (float, optional): learning rate (default: 1e-2)
        inner_steps (integer, optional): number of inner iterations (default: 2)
        delta (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
    """

    def __init__(self, learning_rate=0.01, inner_steps=2, delta=1e-8, name='AdaSFW', **kwargs):
        super().__init__(name, **kwargs)

        self.K = kwargs.get('K', inner_steps)

        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate))
        self._set_hyper('delta', kwargs.get('delta', delta))

    def set_learning_rate(self, learning_rate):
        self._set_hyper('learning_rate', learning_rate)

    def _resource_apply_dense(self, grad, var, constraint, apply_state):
        grad = ops.convert_to_tensor(grad, var.dtype.base_dtype)

        learning_rate = math_ops.cast(self._get_hyper('learning_rate'), var.dtype.base_dtype)
        delta = math_ops.cast(self._get_hyper('delta'), var.dtype.base_dtype)
        accumulator = state_ops.assign_add(self.get_slot(var, "accumulator"), math_ops.square(grad))
        H = math_ops.add(delta, math_ops.sqrt(accumulator))
        y = state_ops.assign(self.get_slot(var, "y"), var)

        for idx in range(self.K):
            delta_q = math_ops.add(grad,
                                   math_ops.multiply(H, math_ops.divide(math_ops.subtract(y, var), learning_rate)))
            v = ops.convert_to_tensor(constraint.lmo(delta_q), var.dtype.base_dtype)
            vy_diff = math_ops.subtract(v, y)
            gamma_unclipped = math_ops.divide(
                math_ops.reduce_sum(- learning_rate * math_ops.multiply(delta_q, vy_diff)),
                math_ops.reduce_sum(math_ops.multiply(H, math_ops.square(vy_diff))))
            gamma = math_ops.ClipByValue(t=gamma_unclipped, clip_value_min=0, clip_value_max=1)
            y = state_ops.assign_add(y, gamma * vy_diff)

        return state_ops.assign(var, y)

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'accumulator',
                          init_ops.constant_initializer(0.0, dtype=var.dtype.base_dtype))  # , initializer="zeros")
            self.add_slot(var, 'y',
                          init_ops.constant_initializer(0.0, dtype=var.dtype.base_dtype))  # , initializer="zeros")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            learning_rate=self._serialize_hyperparameter('learning_rate'),
            delta=self._serialize_hyperparameter('delta'),
        ))
        return config