class RNN_plus_v1_cell(tf.keras.layers.Layer):
    def __init__(self, units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', dropout=0., recurrent_dropout=0., use_bias=True, **kwargs):
        if units < 0:
            raise ValueError(f'Received an invalid value for argument `units`, '
                                f'expected a positive integer, got {units}.')
        # By default use cached variable under v2 mode, see b/143699808.
        if tf.compat.v1.executing_eagerly_outside_functions():
            self._enable_caching_device = kwargs.pop('enable_caching_device', True)
        else:
            self._enable_caching_device = kwargs.pop('enable_caching_device', False)
        super(RNN_plus_v1_cell, self).__init__(**kwargs)
        self.units = units
        self.state_size = self.units
        self.output_size = self.units
        
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.recurrent_initializer = tf.keras.initializers.get(recurrent_initializer)
        self.aux_initializer = tf.keras.initializers.get('zeros')
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.use_bias = True
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units), name='w_input', initializer=self.kernel_initializer, regularizer=None, constraint=None)
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units * 4), name='w_otherpeeps', initializer=self.recurrent_initializer, regularizer=None, constraint=None)
        self.aux  = self.add_weight(shape=(9, 1), name='w_aux', initializer=self.recurrent_initializer, regularizer=None, constraint=None)
        self.bias = self.add_weight( shape=(self.units,), name='b', initializer=self.bias_initializer, regularizer=None, constraint=None) if self.use_bias else None
        self.built = True
        
    def call(self, inputs, states, training=None):
        # prev_output = states[0] if tf.nest.is_nested(states) else states
        prev_output, state0, state1, state2 = states[0], states[1], states[2], states[3]
        
        w_op0, w_op1, w_op2, w_op3 = tf.split(self.recurrent_kernel, num_or_size_splits=4, axis=1)
        w_op0 = tf.linalg.set_diag(w_op0, np.zeros((self.units,), dtype=int))
        w_op1 = tf.linalg.set_diag(w_op1, np.zeros((self.units,), dtype=int))
        w_op2 = tf.linalg.set_diag(w_op2, np.zeros((self.units,), dtype=int))
        w_op3 = tf.linalg.set_diag(w_op3, np.zeros((self.units,), dtype=int))
    
        w_aux = self.aux
                           
        i = tf.keras.backend.dot(inputs, self.kernel) 
        if self.bias is not None:
            i = tf.keras.backend.bias_add(i, self.bias)
        op0 = tf.keras.backend.dot(state0, w_op0)
        op1 = tf.keras.backend.dot(state0, w_op1)
        op2 = tf.keras.backend.dot(state0, w_op2)
        op3 = tf.keras.backend.dot(state0, w_op3)
        
        z = tf.nn.tanh(tf.nn.relu(w_aux[0]*state0 + i) + (w_aux[1]*op0) + (w_aux[2]*state1 + op0))
        iz = 0.5*tf.nn.tanh(op1 + (w_aux[4]*(w_aux[3]*z) + op3) + (w_aux[6]*(w_aux[2]*state1 + w_aux[5]*z + op0) + op1)) + z
        f = 0.5*tf.nn.tanh(prev_output + 0.5*tf.nn.tanh(op2) + 0.5 + w_aux[7]*state1 + w_aux[8]*prev_output + op2) + 0.5 
        s = tf.math.add(tf.keras.backend.dot(f, state2), tf.keras.backend.dot(i, f), name='add_s')
        o = 0.5*tf.nn.tanh(0.5*tf.nn.tanh(s) + 0.5 + w_aux[2]*state1 + op0) + 0.5
        output = o * tf.nn.tanh(s)
        
        return output, [output, z, state0, s]

def rnn_plus_model(noInput, noOutput, timestep):
    """Builds a recurrent model."""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.RNN(cell=RNN_plus_v1_cell(units=hyperparams['noUnits']), input_shape=[timestep, noInput], unroll=False, name='RNNp_layer'))
    model.add(tf.keras.layers.Dense(noInput+noOutput, activation='tanh', name='MLP_layer'))
    model.add(tf.keras.layers.Dense(noOutput))
    optimizer = tf.keras.optimizers.Adam(learning_rate=customLRSchedule(hyperparams['initialLearningRate'], hyperparams['learningRateDecay'], hyperparams['decayDurationFactor'], hyperparams['numTrainingSteps']), \
                                        beta_1=hyperparams['beta1'], beta_2=hyperparams['beta2'], epsilon=hyperparams['epsilon'], amsgrad=False, name="tunedAdam")
    model.compile(optimizer=optimizer, loss = 'mse', run_eagerly=False)
    return model
    

model = rnn_plus_model(noIn, noOut, timestep=40)
model_history = model.fit(
                    x_train, y_train,
                    batch_size=int(hyperparams['batchSize']),
                    verbose=0, # Suppress chatty output; use Tensorboard instead
                    epochs=int(hyperparams['numTrainingSteps']/(x_train.shape[0])),
                    validation_data=(x_val, y_val),
                    shuffle=True,
                    use_multiprocessing=False,
                    callbacks=[tensorboard_callback]
                )
y_pred = model.predict(x_val, verbose=1, batch_size=int(hyperparams['batchSize']))

val_performance = model.evaluate(x_val, y_val, batch_size=1, verbose=1)
y_pred = model.predict(x_val, verbose=1)