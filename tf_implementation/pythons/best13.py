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
        self.aux_initializer = tf.keras.initializers.get('zeros')
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = [self.units, self.units, self.units, self.units]
        self.output_size = self.units
        self.use_bias = True
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units), name='w_input', initializer=self.kernel_initializer, regularizer=None, constraint=None)
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units * 4), name='w_otherpeeps', initializer=self.recurrent_initializer, regularizer=None, constraint=None)
        self.outputs_kernel = self.add_weight(shape=(self.units, self.units * 3), name='w_outputs', initializer=self.recurrent_initializer, regularizer=None, constraint=None)
        self.aux_kernel  = self.add_weight(shape=(9, 1), name='w_aux', initializer=self.recurrent_initializer, regularizer=None, constraint=None)
        self.bias = self.add_weight( shape=(self.units,), name='b', initializer=self.bias_initializer, regularizer=None, constraint=None) if self.use_bias else None
        self.built = True
        
    def call(self, inputs, states, training=None):
        prev_output, state0, state1, state2 = states[0], states[1], states[2], states[3]
        
        # w_in = tf.linalg.set_diag(self.kernel, np.zeros((self.units,), dtype=int))
        w_in = self.kernel
        
        w_op0, w_op1, w_op2, w_op3 = tf.split(self.recurrent_kernel, num_or_size_splits=4, axis=1)
        w_op0 = tf.linalg.set_diag(w_op0, np.zeros((self.units,), dtype=int))
        w_op1 = tf.linalg.set_diag(w_op1, np.zeros((self.units,), dtype=int))
        w_op2 = tf.linalg.set_diag(w_op2, np.zeros((self.units,), dtype=int))
        w_op3 = tf.linalg.set_diag(w_op3, np.zeros((self.units,), dtype=int))
    
        w_aux = self.aux_kernel
        
        w_out0, w_out1, w_out2 = tf.split(self.outputs_kernel, num_or_size_splits=3, axis=1)
        
        inputs = tf.keras.backend.dot(inputs, w_in)
        if self.bias is not None:
            inputs = tf.keras.backend.bias_add(inputs, self.bias)
            
        op0 = tf.keras.backend.dot(state0, w_op0)
        op1 = tf.keras.backend.dot(state0, w_op1)
        op2 = tf.keras.backend.dot(state0, w_op2)
        op3 = tf.keras.backend.dot(state0, w_op3)
        
        p_out0 = tf.keras.backend.dot(prev_output, w_out0)
        p_out1 = tf.keras.backend.dot(prev_output, w_out1)
        p_out2 = tf.keras.backend.dot(prev_output, w_out2)
        
        z = tf.nn.tanh(tf.nn.relu(w_aux[0]*state0 + inputs) + w_aux[1]*op0 + w_aux[2]*state1 + op0)
        i = 0.5*tf.nn.tanh(op1 + w_aux[4]*(w_aux[3]*z + op3) + w_aux[6]*(w_aux[2]*state1 + w_aux[5]*z + op0) + p_out1) + z
        f = 0.5*tf.nn.tanh(prev_output + 0.5*tf.nn.tanh(op2) + 0.5 + w_aux[7]*state1 + w_aux[8]*prev_output + p_out2) + 0.5 
    
        s = tf.math.add(tf.math.multiply(f, state2, name='mul_f_state2'), tf.math.multiply(i, z, name='mul_i_z'), name='add_s')
        o = 0.5*tf.nn.tanh(0.5*tf.nn.tanh(s) + 0.5 + w_aux[2]*state1 + p_out0) + 0.5
        output = o * tf.nn.tanh(s)
        return output, [output, z, state0, s]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return list(_generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype))