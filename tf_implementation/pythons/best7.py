class RNN_plus_v1_cell(tf.keras.layers.LSTMCell):
    def __init__(self, units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', dropout=0., recurrent_dropout=0., use_bias=True, **kwargs):
        if units < 0:
            raise ValueError(f'Received an invalid value for argument `units`, '
                                f'expected a positive integer, got {units}.')
        # By default use cached variable under v2 mode, see b/143699808.
        if tf.compat.v1.executing_eagerly_outside_functions():
            self._enable_caching_device = kwargs.pop('enable_caching_device', True)
        else:
            self._enable_caching_device = kwargs.pop('enable_caching_device', False)
        super(RNN_plus_v1_cell, self).__init__(units, **kwargs)
        self.units = units
        self.state_size = self.units
        self.output_size = self.units
        
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.recurrent_initializer = tf.keras.initializers.get(recurrent_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = [self.units, self.units]
        self.output_size = self.units
        self.use_bias = True
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units), name='w_input', initializer=self.kernel_initializer, regularizer=None, constraint=None)
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units), name='w_otherpeeps', initializer=self.recurrent_initializer, regularizer=None, constraint=None)
        self.outputs_kernel = self.add_weight(shape=(self.units, self.units), name='w_outputs', initializer=self.recurrent_initializer, regularizer=None, constraint=None)
        self.bias = self.add_weight( shape=(self.units,), name='b', initializer=self.bias_initializer, regularizer=None, constraint=None) if self.use_bias else None
        self.built = True
        
    def call(self, inputs, states, training=None):
        prev_output, state0 = states[0], states[1]
        
        # w_in    = tf.linalg.set_diag(self.kernel, np.zeros((self.units,), dtype=int))
        w_in    = self.kernel
        w_out   = tf.linalg.set_diag(self.outputs_kernel, np.zeros((self.units,), dtype=int))
        w_state = tf.linalg.set_diag(self.recurrent_kernel, np.zeros((self.units,), dtype=int))
        
        inputs = tf.keras.backend.dot(inputs, w_in)
        if self.bias is not None:
            inputs = tf.keras.backend.bias_add(inputs, self.bias)
            
        prev_output = tf.keras.backend.dot(prev_output, w_out)
        op0 = tf.keras.backend.dot(state0, w_state)
        
        output = srelu(prev_output - tf.nn.relu(inputs * state0))
        state0 = srelu(tf.nn.relu(inputs) - op0)
        return output, [output, state0]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return list(_generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype))