import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.training.tracking import data_structures

from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin
from tensorflow.python.keras.layers.recurrent import _caching_device
# from tensorflow.python.keras.layers.recurrent import _generate_zero_filled_state_for_cell


class Linear(tf.keras.layers.Layer):
    """Unchanged example from https://www.tensorflow.org/guide/keras/custom_layers_and_models
    Basically a Dense-Layer, nothing new here."""
    def __init__(self, units=32):
        super(Linear, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,)
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="random_normal",
            trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class DenseRNN(tf.keras.layers.Layer):
    """Same as the dense layer above, but for use in an RNN (= with internal state, although unused)"""
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(DenseRNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,)
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="random_normal",
            trainable=True)
        #self.built = True

    def call(self, input_at_t, states_at_t):
        output_at_t = tf.matmul(input_at_t, self.w) + self.b
        states_at_t_plus_1 = output_at_t  # unused
        return output_at_t, states_at_t_plus_1


class IF(tf.keras.layers.Layer):
    """IF layer. Adds input*weight+bias to the internal state.
    Generates spikes as output when state>threshold is reached.
    Reset by substraction."""
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(IF, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,)
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="random_normal",
            trainable=True)
        #self.built = True

    def call(self, input_at_t, states_at_t):
        potential = states_at_t[0] + (tf.matmul(input_at_t, self.w) + self.b)
        output_at_t = tf.cast(tf.math.greater(potential, 1), dtype=tf.float32)
        states_at_t_plus_1 = tf.math.subtract(potential, output_at_t)
        return output_at_t, states_at_t_plus_1


class SpikingReLU(tf.keras.layers.Layer):
    """Roughly the same as the IF above, but without matmul.
    So a standard dense layer without activation has to be used before."""
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(SpikingReLU, self).__init__(**kwargs)

    @tf.autograph.experimental.do_not_convert  # only to suppress warning
    def call(self, input_at_t, states_at_t, training):  # training as keyword only needed with the decorator
        potential = states_at_t[0] + input_at_t
        output_at_t = tf.cast(tf.math.greater(potential, 1), dtype=tf.float32)
        states_at_t_plus_1 = tf.math.subtract(potential, output_at_t)
        return output_at_t, states_at_t_plus_1


class SpikingSigmoid(tf.keras.layers.Layer):
    """Works like the SpikingReLU but is shiftet by 0.5 to the left.
    An neuron with spike adaptation might result in less conversion loss"""
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(SpikingSigmoid, self).__init__(**kwargs)

    @tf.autograph.experimental.do_not_convert  # only to suppress warning
    def call(self, input_at_t, states_at_t, training):  # training as keyword only needed with the decorator
        potential = states_at_t[0] + (input_at_t + 0.5)
        output_at_t = tf.cast(tf.math.greater(potential, 1), dtype=tf.float32)
        states_at_t_plus_1 = tf.math.subtract(potential, output_at_t)
        return output_at_t, states_at_t_plus_1


class SpikingTanh(tf.keras.layers.Layer):
    """Roughly the same as the IF above, but without matmul.
    So a standard dense layer without activation has to be used before."""
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(SpikingTanh, self).__init__(**kwargs)

    @tf.autograph.experimental.do_not_convert  # only to suppress warning
    def call(self, input_at_t, states_at_t, training):  # training as keyword only needed with the decorator
        potential = states_at_t[0] + (input_at_t)
        excitatory = tf.cast(tf.math.greater(potential, 1), dtype=tf.float32)
        inhibitory = -1 * tf.cast(tf.math.less(potential, -1), dtype=tf.float32)
        output_at_t = excitatory + inhibitory
        states_at_t_plus_1 = tf.math.subtract(potential, output_at_t)
        return output_at_t, states_at_t_plus_1


class Accumulate(tf.keras.layers.Layer):
    """Accumulates all input as state for use with a softmax layer."""
    # ToDo: include softmax layer directly here?
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(Accumulate, self).__init__(**kwargs)

    @tf.autograph.experimental.do_not_convert  # only to suppress warning
    def call(self, input_at_t, states_at_t, training):  # training keyword only needed with the decorator
        output_at_t = states_at_t[0] + input_at_t
        states_at_t_plus_1 = output_at_t
        return output_at_t, states_at_t_plus_1


class SpikingLSTM(DropoutRNNCellMixin, Layer):
    def __init__(
            self,
            units,
            #activation='tanh',
            #recurrent_activation='hard_sigmoid',
            use_bias=True,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            unit_forget_bias=True,
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            dropout=0.,
            recurrent_dropout=0.,
            **kwargs):

        # By default use cached variable under v2 mode, see b/143699808.
        if tf.compat.v1.executing_eagerly_outside_functions():
            self._enable_caching_device = kwargs.pop('enable_caching_device', True)
        else:
            self._enable_caching_device = kwargs.pop('enable_caching_device', False)
        super(SpikingLSTM, self).__init__(units, **kwargs)
        self.units = units
        #self.activation = tf.keras.activations.get(activation)
        #self.recurrent_activation = tf.keras.activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.recurrent_initializer = tf.keras.initializers.get(recurrent_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = tf.keras.regularizers.get(recurrent_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)

        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.recurrent_constraint = tf.keras.constraints.get(recurrent_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        """implementation = kwargs.pop('implementation', 1)
        if self.recurrent_dropout != 0 and implementation != 1:
            logging.debug(RECURRENT_DROPOUT_WARNING_MSG)
            self.implementation = 1
        else:
            self.implementation = implementation"""
        self.state_size = data_structures.NoDependency([
            self.units,  # h
            self.units,  # c
            self.units,  # i_state
            self.units,  # f_state
            self.units,  # c_state
            self.units,  # o_state
            self.units  # h_state
        ])
        self.output_size = self.units

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        default_caching_device = _caching_device(self)
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            caching_device=default_caching_device)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            caching_device=default_caching_device)
        self.thresholds = [0.8, 0.9, 1.0, 1.1, 1.2]

        if self.use_bias:
            if self.unit_forget_bias:

                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.get('ones')((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(
                shape=(self.units * 4,),
                name='bias',
                initializer=bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                caching_device=default_caching_device)
        else:
            self.bias = None
        self.built = True

    def _own_sigmoid(self, x):
        return K.sigmoid(x)

    def _own_tanh(self, x):
        return K.tanh(x)

    def _spiking_sigmoid(self, inputs, state):
        potential = state + (inputs + 0.5)
        spikes = tf.cast(tf.math.greater(potential, 1), dtype=tf.float32)
        #state = tf.math.subtract(potential, spikes)  # reset by subtraction
        state = tf.math.subtract(potential, spikes * potential)  # reset to zero
        return spikes, state

    def _spiking_tanh(self, inputs, state):
        potential = state + inputs
        excitatory = tf.cast(tf.math.greater(potential, 0.9), dtype=tf.float32)
        inhibitory = -1 * tf.cast(tf.math.less(potential, -0.9), dtype=tf.float32)
        spikes = excitatory + inhibitory
        #state = tf.math.subtract(potential, spikes)  # reset by subtraction
        state = tf.math.subtract(potential, spikes * potential)  # reset to zero
        return spikes, state

    #def _compute_carry_and_output(self, x, h_tm1, c_tm1):
    #    """Computes carry and output using split kernels."""
    #    x_i, x_f, x_c, x_o = x
    #    h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
    #    i = spiking_sigmoid(x_i + K.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]))
    #    f = spiking_sigmoid(x_f + K.dot(h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]))
    #    c = f * c_tm1 + i * spiking_tanh(x_c + K.dot(h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))
    #    o = spiking_sigmoid(x_o + K.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]))
    #    return c, o

    def _compute_carry_and_output_fused(self, z, c_tm1, i_state, f_state, c_state, o_state):
        """Computes carry and output using fused kernels."""
        z0, z1, z2, z3 = z
        # Original
        #i = self._own_sigmoid(z0)
        #f = self._own_sigmoid(z1)
        #c = f * c_tm1 + i * self._own_tanh(z2)
        #o = self._own_sigmoid(z3)

        # Spiking
        #i, i_state = self._spiking_sigmoid(z0, i_state)
        i = self._own_sigmoid(z0)
        #f, f_state = self._spiking_sigmoid(z1, f_state)
        f = self._own_sigmoid(z1)
        #c_spikes, c_state = self._spiking_tanh(z2, c_state)
        #c = f * c_tm1 + i * c_spikes
        c = f * c_tm1 + i * self._own_tanh(z2)
        #o, o_state = self._spiking_sigmoid(z3, o_state)
        o = self._own_sigmoid(z3)
        return c, o, i_state, f_state, c_state, o_state

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        i_state = states[2]
        f_state = states[3]
        c_state = states[4]
        o_state = states[5]
        h_state = states[6]

        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
        """rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(h_tm1, training, count=4)

        if self.implementation == 1:
            if 0 < self.dropout < 1.:
                inputs_i = inputs * dp_mask[0]
                inputs_f = inputs * dp_mask[1]
                inputs_c = inputs * dp_mask[2]
                inputs_o = inputs * dp_mask[3]
            else:
                inputs_i = inputs
                inputs_f = inputs
                inputs_c = inputs
                inputs_o = inputs
            k_i, k_f, k_c, k_o = tf.split(self.kernel, num_or_size_splits=4, axis=1)
            x_i = K.dot(inputs_i, k_i)
            x_f = K.dot(inputs_f, k_f)
            x_c = K.dot(inputs_c, k_c)
            x_o = K.dot(inputs_o, k_o)
            if self.use_bias:
                b_i, b_f, b_c, b_o = tf.split(self.bias, num_or_size_splits=4, axis=0)
                x_i = K.bias_add(x_i, b_i)
                x_f = K.bias_add(x_f, b_f)
                x_c = K.bias_add(x_c, b_c)
                x_o = K.bias_add(x_o, b_o)

            if 0 < self.recurrent_dropout < 1.:
                h_tm1_i = h_tm1 * rec_dp_mask[0]
                h_tm1_f = h_tm1 * rec_dp_mask[1]
                h_tm1_c = h_tm1 * rec_dp_mask[2]
                h_tm1_o = h_tm1 * rec_dp_mask[3]
            else:
                h_tm1_i = h_tm1
                h_tm1_f = h_tm1
                h_tm1_c = h_tm1
                h_tm1_o = h_tm1
            x = (x_i, x_f, x_c, x_o)
            h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
            c, o = self._compute_carry_and_output(x, h_tm1, c_tm1)
        else:"""
        if 0. < self.dropout < 1.:
            inputs = inputs * dp_mask[0]
        z = K.dot(inputs, self.kernel)
        z += K.dot(h_tm1, self.recurrent_kernel)
        if self.use_bias:
            z = K.bias_add(z, self.bias)

        z = tf.split(z, num_or_size_splits=4, axis=1)
        c, o, i_state, f_state, c_state, o_state = self._compute_carry_and_output_fused(z, c_tm1, i_state, f_state, c_state, o_state)

        #h_spikes, h_state = self._spiking_tanh(c, h_state)
        #h = o * h_spikes
        h = o * self._own_tanh(c)
        return h, [h, c, i_state, f_state, c_state, o_state, h_state]


class PopLSTM(DropoutRNNCellMixin, Layer):
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/layers/recurrent.py#L2252
    def __init__(
            self,
            units,
            n_i=None,
            n_f=None,
            n_c=None,
            n_o=None,
            n_h=None,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            unit_forget_bias=True,
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            #dropout=0.,
            #recurrent_dropout=0.,
            **kwargs):

        if tf.compat.v1.executing_eagerly_outside_functions():
            self._enable_caching_device = kwargs.pop('enable_caching_device', True)
        else:
            self._enable_caching_device = kwargs.pop('enable_caching_device', False)
        super(PopLSTM, self).__init__(units, **kwargs)
        self.units = units
        self.use_bias = use_bias

        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.recurrent_initializer = tf.keras.initializers.get(recurrent_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = tf.keras.regularizers.get(recurrent_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)

        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.recurrent_constraint = tf.keras.constraints.get(recurrent_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        #self.dropout = min(1., max(0., dropout))
        #self.recurrent_dropout = min(1., max(0., recurrent_dropout))

        self.n_i = n_i
        self.n_f = n_f
        self.n_c = n_c
        self.n_o = n_o
        self.n_h = n_h

        self.state_size = data_structures.NoDependency([
            self.units,
            self.units
        ])
        self.output_size = self.units

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        default_caching_device = _caching_device(self)
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            caching_device=default_caching_device)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            caching_device=default_caching_device)

        if self.use_bias:
            if self.unit_forget_bias:

                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.get('ones')((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(
                shape=(self.units * 4,),
                name='bias',
                initializer=bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                caching_device=default_caching_device)
        else:
            self.bias = None
        self.built = True
        if self.n_i is not None:
            self.i_thresh = tf.random.normal([1, self.units, self.n_i], mean=0, stddev=1.75)
        if self.n_f is not None:
            self.f_thresh = tf.random.normal([1, self.units, self.n_f], mean=0, stddev=1.75)
        if self.n_c is not None:
            self.c_thresh_pos = tf.random.normal([1, self.units, self.n_c], mean=0, stddev=0.88)
            self.c_thresh_neg = tf.random.normal([1, self.units, self.n_c], mean=0, stddev=0.88)
        if self.n_o is not None:
            self.o_thresh = tf.random.normal([1, self.units, self.n_o], mean=0, stddev=1.75)
        if self.n_h is not None:
            self.h_thresh_pos = tf.random.normal([1, self.units, self.n_h], mean=0, stddev=0.88)
            self.h_thresh_neg = tf.random.normal([1, self.units, self.n_h], mean=0, stddev=0.88)

    def _spiking_sigmoid(self, inputs, num, thresh):
        expanded = tf.expand_dims(inputs, axis=2)
        repeated = tf.repeat(expanded, repeats=num, axis=2)
        spikes = tf.cast(tf.math.greater(repeated, thresh), dtype=tf.float32)
        reduce_sum = tf.math.reduce_sum(spikes, axis=2)
        spiking_sigmoid = reduce_sum / num
        return spiking_sigmoid

    def _spiking_tanh(self, inputs, num, thresh_pos, thresh_neg):
        expanded = tf.expand_dims(inputs, axis=2)
        repeated = tf.repeat(expanded, repeats=num, axis=2)
        # stacked = tf.stack([inputs]*num, axis=2)  # ~20% slower
        excitatory = tf.cast(tf.math.greater(repeated, thresh_pos), dtype=tf.float32)
        inhibitory = -1 * tf.cast(tf.math.less(repeated, thresh_neg), dtype=tf.float32)
        spikes = excitatory + inhibitory
        reduce_sum = tf.math.reduce_sum(spikes, axis=2)
        spiking_tanh = reduce_sum / num
        return spiking_tanh

    def _compute_carry_and_output_fused(self, z, c_tm1):  # , i_state, f_state, c_state, o_state):
        """Computes carry and output using fused kernels."""
        z0, z1, z2, z3 = z

        # Input Gate
        if self.n_i is None:
            i = K.sigmoid(z0)
        else:
            i = self._spiking_sigmoid(z0, self.n_i, self.i_thresh)

        # Forget Gate
        if self.n_f is None:
            f = K.sigmoid(z1)
        else:
            f = self._spiking_sigmoid(z1, self.n_f, self.f_thresh)

        # Cell State
        if self.n_c is None:
            c = f * c_tm1 + i * K.tanh(z2)
        else:
            c_tanh = self._spiking_tanh(z2, self.n_c, self.c_thresh_pos, self.c_thresh_neg)
            c = (f * c_tm1) + (i * c_tanh)

        # Output Gate
        if self.n_o is None:
            o = K.sigmoid(z3)
        else:
            o = self._spiking_sigmoid(z3, self.n_o, self.o_thresh)
        return c, o  # , i_state, f_state, c_state, o_state

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        z = K.dot(inputs, self.kernel)
        z += K.dot(h_tm1, self.recurrent_kernel)
        if self.use_bias:
            z = K.bias_add(z, self.bias)

        z = tf.split(z, num_or_size_splits=4, axis=1)
        #c, o, i_state, f_state, c_state, o_state = self._compute_carry_and_output_fused(z, c_tm1, i_state, f_state, c_state, o_state)
        c, o = self._compute_carry_and_output_fused(z, c_tm1)

        if self.n_h is None:
            h = o * K.tanh(c)
        else:
            h = o * self._spiking_tanh(c, self.n_h, self.h_thresh_pos, self.h_thresh_neg)
        return h, [h, c]  # , i_state, f_state, c_state, o_state, h_state]


class PopGRUCell(DropoutRNNCellMixin, Layer):
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/layers/recurrent.py#L1692
  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               reset_after=False,
               **kwargs):
    if units < 0:
      raise ValueError(f'Received an invalid value for units, expected '
                       f'a positive integer, got {units}.')
    # By default use cached variable under v2 mode, see b/143699808.
    if ops.executing_eagerly_outside_functions():
      self._enable_caching_device = kwargs.pop('enable_caching_device', True)
    else:
      self._enable_caching_device = kwargs.pop('enable_caching_device', False)
    super(PopGRUCell, self).__init__(**kwargs)
    self.units = units
    self.activation = activations.get(activation)
    self.recurrent_activation = activations.get(recurrent_activation)
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)
    self.bias_initializer = initializers.get(bias_initializer)

    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)

    self.kernel_constraint = constraints.get(kernel_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.dropout = min(1., max(0., dropout))
    self.recurrent_dropout = min(1., max(0., recurrent_dropout))

    implementation = kwargs.pop('implementation', 1)
    if self.recurrent_dropout != 0 and implementation != 1:
      logging.debug(RECURRENT_DROPOUT_WARNING_MSG)
      self.implementation = 1
    else:
      self.implementation = implementation
    self.reset_after = reset_after
    self.state_size = self.units
    self.output_size = self.units

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    input_dim = input_shape[-1]
    default_caching_device = _caching_device(self)
    self.kernel = self.add_weight(
        shape=(input_dim, self.units * 3),
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        caching_device=default_caching_device)
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units * 3),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint,
        caching_device=default_caching_device)

    if self.use_bias:
      if not self.reset_after:
        bias_shape = (3 * self.units,)
      else:
        # separate biases for input and recurrent kernels
        # Note: the shape is intentionally different from CuDNNGRU biases
        # `(2 * 3 * self.units,)`, so that we can distinguish the classes
        # when loading and converting saved weights.
        bias_shape = (2, 3 * self.units)
      self.bias = self.add_weight(shape=bias_shape,
                                  name='bias',
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint,
                                  caching_device=default_caching_device)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs, states, training=None):
    h_tm1 = states[0] if nest.is_nested(states) else states  # previous memory

    dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=3)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        h_tm1, training, count=3)

    if self.use_bias:
      if not self.reset_after:
        input_bias, recurrent_bias = self.bias, None
      else:
        input_bias, recurrent_bias = array_ops.unstack(self.bias)

    if self.implementation == 1:
      if 0. < self.dropout < 1.:
        inputs_z = inputs * dp_mask[0]
        inputs_r = inputs * dp_mask[1]
        inputs_h = inputs * dp_mask[2]
      else:
        inputs_z = inputs
        inputs_r = inputs
        inputs_h = inputs

      x_z = backend.dot(inputs_z, self.kernel[:, :self.units])
      x_r = backend.dot(inputs_r, self.kernel[:, self.units:self.units * 2])
      x_h = backend.dot(inputs_h, self.kernel[:, self.units * 2:])

      if self.use_bias:
        x_z = backend.bias_add(x_z, input_bias[:self.units])
        x_r = backend.bias_add(x_r, input_bias[self.units: self.units * 2])
        x_h = backend.bias_add(x_h, input_bias[self.units * 2:])

      if 0. < self.recurrent_dropout < 1.:
        h_tm1_z = h_tm1 * rec_dp_mask[0]
        h_tm1_r = h_tm1 * rec_dp_mask[1]
        h_tm1_h = h_tm1 * rec_dp_mask[2]
      else:
        h_tm1_z = h_tm1
        h_tm1_r = h_tm1
        h_tm1_h = h_tm1

      recurrent_z = backend.dot(h_tm1_z, self.recurrent_kernel[:, :self.units])
      recurrent_r = backend.dot(
          h_tm1_r, self.recurrent_kernel[:, self.units:self.units * 2])
      if self.reset_after and self.use_bias:
        recurrent_z = backend.bias_add(recurrent_z, recurrent_bias[:self.units])
        recurrent_r = backend.bias_add(
            recurrent_r, recurrent_bias[self.units:self.units * 2])

      z = self.recurrent_activation(x_z + recurrent_z)
      r = self.recurrent_activation(x_r + recurrent_r)

      # reset gate applied after/before matrix multiplication
      if self.reset_after:
        recurrent_h = backend.dot(
            h_tm1_h, self.recurrent_kernel[:, self.units * 2:])
        if self.use_bias:
          recurrent_h = backend.bias_add(
              recurrent_h, recurrent_bias[self.units * 2:])
        recurrent_h = r * recurrent_h
      else:
        recurrent_h = backend.dot(
            r * h_tm1_h, self.recurrent_kernel[:, self.units * 2:])

      hh = self.activation(x_h + recurrent_h)
    else:
      if 0. < self.dropout < 1.:
        inputs = inputs * dp_mask[0]

      # inputs projected by all gate matrices at once
      matrix_x = backend.dot(inputs, self.kernel)
      if self.use_bias:
        # biases: bias_z_i, bias_r_i, bias_h_i
        matrix_x = backend.bias_add(matrix_x, input_bias)

      x_z, x_r, x_h = array_ops.split(matrix_x, 3, axis=-1)

      if self.reset_after:
        # hidden state projected by all gate matrices at once
        matrix_inner = backend.dot(h_tm1, self.recurrent_kernel)
        if self.use_bias:
          matrix_inner = backend.bias_add(matrix_inner, recurrent_bias)
      else:
        # hidden state projected separately for update/reset and new
        matrix_inner = backend.dot(
            h_tm1, self.recurrent_kernel[:, :2 * self.units])

      recurrent_z, recurrent_r, recurrent_h = array_ops.split(
          matrix_inner, [self.units, self.units, -1], axis=-1)

      z = self.recurrent_activation(x_z + recurrent_z)
      r = self.recurrent_activation(x_r + recurrent_r)

      if self.reset_after:
        recurrent_h = r * recurrent_h
      else:
        recurrent_h = backend.dot(
            r * h_tm1, self.recurrent_kernel[:, 2 * self.units:])

      hh = self.activation(x_h + recurrent_h)
    # previous and candidate state mixed by update gate
    h = z * h_tm1 + (1 - z) * hh
    new_state = [h] if nest.is_nested(states) else h
    return h, new_state

  def get_config(self):
    config = {
        'units': self.units,
        'activation': activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias': self.use_bias,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint),
        'dropout': self.dropout,
        'recurrent_dropout': self.recurrent_dropout,
        'implementation': self.implementation,
        'reset_after': self.reset_after
    }
    config.update(_config_for_enable_caching_device(self))
    base_config = super(GRUCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    return _generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype)

