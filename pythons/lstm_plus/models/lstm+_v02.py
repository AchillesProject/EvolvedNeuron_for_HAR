import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.util.tf_export import keras_export
from tensorflow.keras.backend import eval

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, RobustScaler, StandardScaler

# import tensorboard
import keras
from keras.utils import tf_utils
import pandas as pd #pd.plotting.register_matplotlib_converters
import numpy as np
import sys, os, math, time, datetime, re

print("tf: ", tf.__version__)
# print("tb: ", tensorboard.__version__)
print(os.getcwd())

DTYPE = tf.float64
RANDOM_SEED = 42
ISMOORE_DATASETS = True
timestep = 40
tf.random.set_seed(np.random.seed(RANDOM_SEED))
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(1)
tf.config.set_visible_devices([], 'GPU')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

tf.keras.backend.set_floatx('float64')

with open("../params/params_har.txt") as f:
    hyperparams = dict([re.sub('['+' ,\n'+']','',x.replace(' .', '')).split('=') for x in f][1:-1])
hyperparams = dict([k, float(v)] for k, v in hyperparams.items())
hyperparams['testSize'] = 0.500
hyperparams['noUnits'] = 81
hyperparams['timestep'] = 40
print(hyperparams)

def seperateValues(data, noInput, noOutput, isMoore=True):
    x_data, y_data = None, None
    for i in range(data.shape[0]):
        if isMoore:
            x_data_i = data[i].reshape(-1, noInput+noOutput)
            x_data_i, y_data_i = x_data_i[:, 0:noInput], x_data_i[-1, noInput:]
        else:
            x_data_i = data[i][:-1].reshape(-1, noInput)
            y_data_i = data[i][-1].reshape(-1, noOutput)
        x_data = x_data_i[np.newaxis,:,:] if x_data is None else np.append(x_data, x_data_i[np.newaxis,:,:], axis=0)
        y_data = y_data_i.reshape(1, -1) if y_data is None else np.append(y_data, y_data_i.reshape(1, -1), axis=0)
    return x_data, y_data

def fromBit_v0( b ) :
    return -0.9 if b == 0.0 else 0.9

def fromBit_v1( b ) :
    return 0 if b == 0.0 else 1

def isCorrect( target, actual ) :
    y1 = False if target < 0.0 else True
    y2 = False if actual < 0.0 else True
    return y1 == y2 

class CustomMetricError(tf.keras.metrics.MeanMetricWrapper):
    def __init__(self, name='custom_metric_error', dtype=None, threshold=0.5):
        super(CustomMetricError, self).__init__(
            customMetricfn_tensor, name, dtype=dtype, threshold=threshold)

    def customMetricfn_tensor(true, pred, threshold=0.5):
        true = tf.convert_to_tensor(true)
        pred = tf.convert_to_tensor(pred)
        threshold = tf.cast(threshold, pred.dtype)
        pred = tf.cast(pred >= threshold, pred.dtype)
        true = tf.cast(true >= threshold, true.dtype)
        return keras.backend.mean(tf.equal(true, pred), axis=-1)

def customMetricfn(y_true, y_pred):
    numCorrect = 0
    for i in range( y_true.shape[0] ) :
        for j in range( y_pred.shape[ 1 ] ) :
            if isCorrect( y_true[ i, j ], y_pred[ i, j ] ) :
                numCorrect += 1
    return (numCorrect/(y_pred.shape[1]*y_true.shape[0]))

def indexOfMax( xs ) :
    m, k = xs[ 0 ], 0
    for i in range( 0, xs.size ) :
        if xs[ i ] > m :
            m = xs[ i ]
            k = i
    return k

def customMetricfn_full(y_true, y_pred):
    numCorrect = 0
    for i in range(y_pred.shape[0]) :
        if indexOfMax(y_pred[i]) == indexOfMax(y_val[i]) :
            numCorrect += 1
    return numCorrect/y_pred.shape[0]
            
def srelu(x):
    return tf.keras.backend.clip(x, -1, 1)

def _generate_zero_filled_state_for_cell(cell, inputs, batch_size, dtype):
    if inputs is not None:
        batch_size = tf.shape(inputs)[0]
        dtype = inputs.dtype
    return _generate_zero_filled_state(batch_size, cell.state_size, dtype)

def _generate_zero_filled_state(batch_size_tensor, state_size, dtype):
    def create_zeros(unnested_state_size):
        flat_dims = tf.TensorShape(unnested_state_size).as_list()
        init_state_size = [batch_size_tensor] + flat_dims
        return tf.zeros(init_state_size, dtype=dtype)
    
    if batch_size_tensor is None or dtype is None:
        raise ValueError(
            'batch_size and dtype cannot be None while constructing initial state. '
            f'Received: batch_size={batch_size_tensor}, dtype={dtype}')

    return tf.nest.map_structure(create_zeros, state_size)  if tf.nest.is_nested(state_size) else create_zeros(state_size)

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
        self.state_size = [self.units, self.units, self.units, self.units, self.units]
        self.output_size = self.units
        self.use_bias = True
        self.cell_dtype = DTYPE
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units * 2), name='w_input', initializer=self.kernel_initializer, regularizer=None, constraint=None, dtype=self.cell_dtype)
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units * 4), name='w_otherpeeps', initializer=self.recurrent_initializer, regularizer=None, constraint=None, dtype=self.cell_dtype)
        self.aux_kernel  = self.add_weight(shape=(5, self.units), name='w_aux', initializer=self.recurrent_initializer, regularizer=None, constraint=None, dtype=self.cell_dtype)
        self.bias = self.add_weight( shape=(self.units,), name='b', initializer=self.bias_initializer, regularizer=None, constraint=None, dtype=self.cell_dtype) if self.use_bias else None
        self.built = True
        
    def call(self, inputs, states, training=None):
        state0, state1, state2, state3, prev_output = states[0], states[1], states[2], states[3], states[4]
        
        w_in_0, w_in_2 = tf.split(self.kernel, num_or_size_splits=2, axis=1)

        w_op0, w_op2, w_op3, w_op4 = tf.split(self.recurrent_kernel, num_or_size_splits=4, axis=1)
        w_op0 = tf.linalg.set_diag(w_op0, np.zeros((self.units,), dtype=np.float64))
        w_op2 = tf.linalg.set_diag(w_op2, np.zeros((self.units,), dtype=np.float64))
        w_op3 = tf.linalg.set_diag(w_op3, np.zeros((self.units,), dtype=np.float64))
        w_op4 = tf.linalg.set_diag(w_op4, np.zeros((self.units,), dtype=np.float64))
    
        w_aux = self.aux_kernel
        
        inputs_0 = tf.keras.backend.dot(inputs, w_in_0)
        # inputs_1 = tf.keras.backend.dot(inputs, w_in_1)
        inputs_2 = tf.keras.backend.dot(inputs, w_in_2)
        
        if self.bias is not None:
            inputs_0 = tf.keras.backend.bias_add(inputs_0, self.bias)
            # inputs_1 = tf.keras.backend.bias_add(inputs_1, self.bias)
            inputs_2 = tf.keras.backend.bias_add(inputs_2, self.bias)
            
        op0 = tf.keras.backend.dot(state0, w_op0)
        # op1 = tf.keras.backend.dot(state0, w_op1)
        op2 = tf.keras.backend.dot(state0, w_op2)
        op3 = tf.keras.backend.dot(state0, w_op3)
        op4 = tf.keras.backend.dot(state0, w_op4)
        
        z1 = tf.nn.tanh(tf.nn.tanh(tf.nn.tanh(op4*tf.nn.tanh(srelu(w_aux[0]*op3 + inputs_0)))))
        z2 = tf.nn.tanh(tf.nn.tanh(tf.nn.tanh(srelu(tf.nn.tanh(w_aux[1]*op2 + w_aux[2]*state3 + w_aux[3])))))
        z3 = tf.nn.tanh(tf.nn.relu(inputs_2))
        z  = z1 - (z2 + z3)
        output = prev_output - (z - state1)*z

        return output, [z, state0, w_aux[4]*z + op0, state2, output]
    
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return list(_generate_zero_filled_state_for_cell(self, inputs, batch_size, dtype))

class LearningRateLoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print("Learning Rate: ", self.model.optimizer.learning_rate.lr)
    
    def on_epoch_end(self, epoch, logs=None):
        tf.summary.scalar('learning rate', self.model.optimizer.learning_rate.lr, step=epoch)

class customLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, batchSize, initialLearningRate, learningRateDecay, 
                 decayDurationFactor, numTrainingSteps, name=None):
        self.name                 = name
        self.cell_dtype           = DTYPE
        self.batchSize            = tf.constant(batchSize, dtype=self.cell_dtype, name="bz")
        self.initialLearningRate  = tf.constant(initialLearningRate, dtype=self.cell_dtype, name="lr0") 
        self.learningRateDecay    = tf.constant(learningRateDecay, dtype=self.cell_dtype, name="alpha")
        self.decayDurationFactor  = tf.constant(decayDurationFactor, dtype=self.cell_dtype, name="beta")
        self.numTrainingSteps     = tf.constant(numTrainingSteps, dtype=self.cell_dtype, name="ortho")
        self.T                    = tf.constant(self.decayDurationFactor*(self.numTrainingSteps/self.batchSize), 
                                                dtype=self.cell_dtype, name="T")
        self.lr                   = tf.Variable(self.initialLearningRate, dtype=self.cell_dtype, name="lr")
    
    def __call__(self, step):
        self.t = tf.cast(step, self.cell_dtype)
        self.lr = tf.cond(self.t > self.T, 
           lambda: self.learningRateDecay * self.initialLearningRate,
           lambda: self.initialLearningRate -(1.0-self.learningRateDecay)*self.initialLearningRate*self.t/self.T
          )
        return self.lr
    
    def get_config(self):
        return {
            "name":           self.name,
            "cell_dtype":     self.cell_dtype,
            "batchSize":      self.batchSize,
            "initial_lr":     self.initialLearningRate,
            "decay_rate":     self.learningRateDecay,
            "decay_duration": self.decayDurationFactor,
            "training_step":  self.numTrainingSteps,
            "curr_lr":        self.lr
        }
    
def rnn_plus_model(noInput, noOutput, timestep):
    """Builds a recurrent model."""
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.RNN(cell=RNN_plus_v1_cell(units=hyperparams['noUnits']), input_shape=[timestep, noInput], unroll=False, name='RNNp_layer', dtype=DTYPE))
    model.add(tf.keras.layers.Dense(noInput+noOutput, activation='tanh', name='MLP_layer'))
    model.add(tf.keras.layers.Dense(noOutput, name='Output_layer'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=customLRSchedule(hyperparams['batchSize'], hyperparams['initialLearningRate'], hyperparams['learningRateDecay'], hyperparams['decayDurationFactor'], hyperparams['numTrainingSteps']), \
                                        beta_1=hyperparams['beta1'], beta_2=hyperparams['beta2'], epsilon=hyperparams['epsilon'], amsgrad=False, name="tunedAdam")
    model.compile(optimizer=optimizer, loss = 'mse', run_eagerly=False)
    print(tf.keras.backend.floatx())
    print(model.summary())
    return model


class LSTMCell_modified(tf.keras.layers.LSTMCell):
    def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
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
        if units < 0:
            raise ValueError(f'Received an invalid value for argument `units`, '
                           f'expected a positive integer, got {units}.')
        # By default use cached variable under v2 mode, see b/143699808.
        if tf.compat.v1.executing_eagerly_outside_functions():
            self._enable_caching_device = kwargs.pop('enable_caching_device', True)
        else:
            self._enable_caching_device = kwargs.pop('enable_caching_device', False)
        super(LSTMCell_modified, self).__init__(units, **kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.recurrent_activation = tf.keras.activations.get(recurrent_activation)
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
        implementation = kwargs.pop('implementation', 1)
        if self.recurrent_dropout != 0 and implementation != 1:
            logging.debug(RECURRENT_DROPOUT_WARNING_MSG)
            self.implementation = 1
        else:
            self.implementation = implementation
        self.state_size = [self.units, self.units, self.units]
        self.output_size = self.units

    def build(self, input_shape):
        # default_caching_device = _caching_device(self)
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 4),
            name='kernel',
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint   
            # caching_device=default_caching_device
        )
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint
            # caching_device=default_caching_device
        )

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(_, *args, **kwargs):
                    return tf.keras.backend.concatenate([
                      self.bias_initializer((self.units,), *args, **kwargs),
                      tf.keras.initializers.get('ones')((self.units,), *args, **kwargs),
                      self.bias_initializer((self.units * 2,), *args, **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(
                  shape=(self.units * 4,),
                  name='bias',
                  initializer=bias_initializer,
                  regularizer=self.bias_regularizer,
                  constraint=self.bias_constraint
                  # caching_device=default_caching_device
            )
        else:
            self.bias = None
        self.built = True

    def _compute_carry_and_output(self, x, h_tm1, c_tm1):
        """Computes carry and output using split kernels."""
        x_i, x_f, x_c, x_o = x
        h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
        i = self.recurrent_activation(
            x_i + tf.keras.backend.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]))
        f = self.recurrent_activation(x_f + tf.keras.backend.dot(
            h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]))
        c = f * c_tm1 + i * self.activation(x_c + tf.keras.backend.dot(
            h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))
        o = self.recurrent_activation(
            x_o + tf.keras.backend.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]))
        return c, o

    def _compute_carry_and_output_fused(self, z, c_tm1):
        """Computes carry and output using fused kernels."""
        z0, z1, z2, z3 = z
        i = self.recurrent_activation(z0)
        f = self.recurrent_activation(z1)
        c = f * c_tm1 + i * self.activation(z2)
        o = self.recurrent_activation(z3)
        return c, o

    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        c_tm2 = states[2]  # second previous carry state

        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training, count=4)

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
            k_i, k_f, k_c, k_o = tf.split(
                self.kernel, num_or_size_splits=4, axis=1)
            x_i = tf.keras.backend.dot(inputs_i, k_i)
            x_f = tf.keras.backend.dot(inputs_f, k_f)
            x_c = tf.keras.backend.dot(inputs_c, k_c)
            x_o = tf.keras.backend.dot(inputs_o, k_o)
            if self.use_bias:
                b_i, b_f, b_c, b_o = tf.split(
                    self.bias, num_or_size_splits=4, axis=0)
                x_i = tf.keras.backend.bias_add(x_i, b_i)
                x_f = tf.keras.backend.bias_add(x_f, b_f)
                x_c = tf.keras.backend.bias_add(x_c, b_c)
                x_o = tf.keras.backend.bias_add(x_o, b_o)

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
        else:
            if 0. < self.dropout < 1.:
                inputs = inputs * dp_mask[0]
            z = tf.keras.backend.dot(inputs, self.kernel)
            z += tf.keras.backend.dot(h_tm1, self.recurrent_kernel)
            if self.use_bias:
                z = tf.keras.backend.bias_add(z, self.bias)

            z = tf.split(z, num_or_size_splits=4, axis=1)
            c, o = self._compute_carry_and_output_fused(z, c_tm2)

        h = o * self.activation(c)
        return h, [h, c, c_tm1]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return list(_generate_zero_filled_state_for_cell(
            self, inputs, batch_size, dtype))


class LSTM_modified(tf.keras.layers.RNN):
    def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               **kwargs):
        implementation = kwargs.pop('implementation', 1)
        if implementation == 0:
            logging.warning('`implementation=0` has been deprecated, '
                          'and now defaults to `implementation=1`.'
                          'Please update your layer call.')
        if 'enable_caching_device' in kwargs:
            cell_kwargs = {'enable_caching_device':
                         kwargs.pop('enable_caching_device')}
        else:
            cell_kwargs = {}
        cell = LSTMCell_modified(
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            unit_forget_bias=unit_forget_bias,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            implementation=implementation,
            dtype=kwargs.get('dtype'),
            trainable=kwargs.get('trainable', True),
            **cell_kwargs)
        super(LSTM_modified, self).__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.input_spec = [tf.keras.layers.InputSpec(ndim=3)]

    def call(self, inputs, mask=None, training=None, initial_state=None):
        return super(LSTM_modified, self).call(
            inputs, mask=mask, training=training, initial_state=initial_state)

    
def lstm_modified_model(noInput, noOutput, timestep):
    """Builds a recurrent model."""
    
    model = tf.keras.Sequential()
    model.add(LSTM_modified(units=int(hyperparams['noUnits']), input_shape=[timestep, noInput],
                   activation='tanh', recurrent_activation='sigmoid', unroll=False, use_bias=True,
                   recurrent_dropout=0.0, return_sequences=False, name='LSTM_layer'))
    model.add(tf.keras.layers.Dense(noInput+noOutput, activation='tanh', name='MLP_layer'))
    model.add(tf.keras.layers.Dense(noOutput, name='Output_layer'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=customLRSchedule(hyperparams['batchSize'], hyperparams['initialLearningRate'], hyperparams['learningRateDecay'], hyperparams['decayDurationFactor'], hyperparams['numTrainingSteps']), \
                                        beta_1=hyperparams['beta1'], beta_2=hyperparams['beta2'], epsilon=hyperparams['epsilon'], amsgrad=False, name="tunedAdam")
    model.compile(optimizer=optimizer, loss = 'mse', run_eagerly=False)
    print(tf.keras.backend.floatx())
    print(model.summary())
    return model

#===============MAIN=================
if __name__ == '__main__':
    ISMOORE_DATASETS = True
    noIn, noOut = 3, 6
    path = '../../Datasets/6_har/0_WISDM/WISDM_ar_v1.1/wisdm_script_and_data/wisdm_script_and_data/WISDM/testdata/' #fulla node1 path
    fileslist = [f for f in sorted(os.listdir(path)) if os.path.isfile(os.path.join(path, f))]
    # logdir = f"./logs/scalars/wisdm"
    # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    # print(hyperparams)
    print(tf.keras.backend.floatx())
    for file_no in range(8):
        trainFile = f'train{file_no}.csv'
        valFile   = f'val{file_no}.csv'
        df_train  = np.array(pd.read_csv(os.path.join(path, trainFile), skiprows=1))
        df_val    = np.array(pd.read_csv(os.path.join(path, valFile), skiprows=1))

        scaler    = StandardScaler()
        x_train, y_train = seperateValues(df_train, noIn, noOut, isMoore=ISMOORE_DATASETS)
        x_val,   y_val   = seperateValues(df_val,   noIn, noOut, isMoore=ISMOORE_DATASETS) 
        x_train   = (scaler.fit_transform(x_train.reshape(x_train.shape[0], -1))).reshape(x_train.shape[0], hyperparams['timestep'], noIn)
        x_val     = (scaler.fit_transform(x_val.reshape(x_val.shape[0], -1))).reshape(x_val.shape[0], hyperparams['timestep'], noIn)
        for i in range( y_train.shape[ 0 ]) :
            for j in range( y_train.shape[1]) :
                y_train[i, j] = fromBit_v1(y_train[i,j])
        for i in range(y_val.shape[0]):
            for j in range(y_val.shape[ 1 ]):
                y_val[i, j] = fromBit_v1(y_val[ i, j ])

        model = lstm_modified_model(noIn, noOut, timestep=hyperparams['timestep'])
        model_history = model.fit(
                            x_train, y_train,
                            batch_size=int(hyperparams['batchSize']),
                            verbose=1, # Suppress chatty output; use Tensorboard instead
                            epochs=int(hyperparams['numTrainingSteps']/(x_train.shape[0])),
                            validation_data=(x_val, y_val),
                            shuffle=True,
                            use_multiprocessing=True,
                            #callbacks=[tensorboard_callback, LearningRateLoggingCallback()],
                        )
        y_pred = model.predict(x_val, verbose=0, batch_size=int(hyperparams['batchSize']))
        val_performance = model.evaluate(x_val, y_val, batch_size=int(hyperparams['batchSize']), verbose=0)
        print(f"{valFile} val_performance = {val_performance}")
        print(f"{valFile} val accuracy = {round(customMetricfn_full(y_val, y_pred), 5)}")
