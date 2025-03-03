import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, LSTM, MaxPooling2D,AveragePooling2D,GlobalMaxPooling2D, GlobalAveragePooling2D, Flatten, Dropout, Reshape, BatchNormalization, ReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.python.util.tf_export import keras_export
from tensorflow.keras.backend import eval

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, RobustScaler

import tensorboard
import keras
from keras.utils import tf_utils

from functools import partial
from matplotlib import rc, style
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
import pandas as pd #pd.plotting.register_matplotlib_converters
import numpy as np
from scipy import stats

import sys, os, math, time, datetime, re

print("tf: ", tf.__version__)
print("tb: ", tensorboard.__version__)
print(os.getcwd())

%matplotlib inline
%config InlineBackend.figure_format='retina'

style.use("seaborn")
pd.plotting.register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale = 1)

# rcParams['figure.figsize'] = 22, 10

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(1)
tf.config.set_visible_devices([], 'GPU')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
tf.keras.backend.set_floatx('float64')

# Debugging with Tensorboard
snapshot = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir="logs/fit/rnn_v1_1/" + snapshot

path = "../../Datasets/1_5_big_datasets/bigdatasets" #"../../Datasets/0_180_small_datasets/Version9.128timesteps"
fileslist = [f for f in sorted(os.listdir(path)) if os.path.isfile(os.path.join(path, f))]

with open("./params/params_5_bigsets.txt") as f:
    hyperparams = dict([re.sub('['+' ,\n'+']','',x.replace(' .', '')).split('=') for x in f][1:-1])
hyperparams = dict([k, float(v)] for k, v in hyperparams.items())
hyperparams['testSize'] = 0.5
display(hyperparams)

def seperateValues(data, noInput, noOutput):
    x_data, y_data = None, None
    for i in range(data.shape[0]):
        x_data_i = data[i].reshape(-1, noInput+noOutput).astype('float64')
        x_data_i, y_data_i = x_data_i[:, 0:noInput], x_data_i[-1, noInput:]
        x_data = x_data_i[np.newaxis,:,:] if x_data is None else np.append(x_data, x_data_i[np.newaxis,:,:], axis=0)
        y_data = y_data_i.reshape(1, -1) if y_data is None else np.append(y_data, y_data_i.reshape(1, -1), axis=0)
    return x_data, y_data

def fromBit( b ) :
    return -0.9 if b == 0.0 else 0.9

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
    count, numCorrect = 0, 0
    for i in range( y_true.shape[0] ) :
        for j in range( y_pred.shape[ 1 ] ) :
            count += 1
            if isCorrect( y_true[ i, j ], y_pred[ i, j ] ) :
                numCorrect += 1
    return (numCorrect/count)

def isCorrect( target, actual ) :
    y1 = False if target < 0.0 else True
    y2 = False if actual < 0.0 else True
    return y1 == y2 

class customLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initialLearningRate, learningRateDecay, decayDurationFactor, numTrainingSteps, glorotScaleFactor=0.1, orthogonalScaleFactor=0.1, name=None):
        self.initialLearningRate = initialLearningRate
        self.learningRateDecay = learningRateDecay
        self.decayDurationFactor = decayDurationFactor
        self.glorotScaleFactor = glorotScaleFactor
        self.orthogonalScaleFactor = orthogonalScaleFactor
        self.numTrainingSteps = numTrainingSteps
        self.name = name
        self.T = tf.constant(self.decayDurationFactor * self.numTrainingSteps, dtype=tf.float32, name="T")
    
    def __call__(self, step):
        self.step = tf.cast(step, tf.float32)
        self.lr = tf.cond(self.step > self.T, 
                           lambda: tf.constant(self.learningRateDecay * self.initialLearningRate, dtype=tf.float32),
                           lambda: self.initialLearningRate * (1.0 - (1.0 - self.learningRateDecay) * self.step / self.T)
                          )
        return self.lr
    
class RNN_plus_v1_cell(keras.engine.base_layer.Layer):
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
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.use_bias = True
    
    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units), name='w_i', initializer=self.kernel_initializer, regularizer=None, constraint=None)
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units), name='w_o', initializer=self.recurrent_initializer, regularizer=None, constraint=None)
        self.bias = self.add_weight( shape=(self.units,), name='b', initializer=self.bias_initializer, regularizer=None, constraint=None) if self.use_bias else None
        self.built = True
        
    def call(self, inputs, states, training=None):
        prev_output = states[0] if tf.nest.is_nested(states) else states
        i = tf.keras.backend.dot(inputs, self.kernel)
        
        if self.bias is not None:
            i = tf.keras.backend.bias_add(i, self.bias)

        z = tf.keras.backend.dot(prev_output, tf.linalg.set_diag(self.recurrent_kernel, np.zeros((self.units,), dtype=int)))
        iz = tf.math.add(i, z, name='add_iz')
        v = tf.math.subtract(tf.math.square(iz,name='square_iz'), iz, name='sub_v')
        output = tf.keras.backend.clip(v, -1, 1)

        new_state = [output] if tf.nest.is_nested(states) else output
        return output, new_state

def rnn_plus_model(timestep, noInput, noOutput, batchSize):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(timestep, noInput), name='Input_layer'))
    model.add(tf.keras.layers.RNN(cell=RNN_plus_v1_cell(units=noInput+noOutput), unroll=False, name='RNNp_layer'))
    model.add(tf.keras.layers.Dense(noOutput,'tanh', name='MLP_layer'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=customLRSchedule(hyperparams['initialLearningRate'], hyperparams['learningRateDecay'], hyperparams['decayDurationFactor'], hyperparams['numTrainingSteps']), \
                                        beta_1=hyperparams['beta1'], beta_2=hyperparams['beta2'], epsilon=hyperparams['epsilon'], amsgrad=False, name="tunedAdam")
    model.compile(optimizer=optimizer, loss = 'mse', metrics=[CustomMetricError(threshold=0.0)], run_eagerly=False)
    return model

def rnn_model(timestep, noInput, noOutput, batchSize):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(timestep, noInput),name='Input_layer'))
    model.add(tf.keras.layers.RNN(cell=tf.keras.layers.SimpleRNNCell(units=noInput+noOutput), name='SimpleRNN_layer', stateful=False))
    model.add(tf.keras.layers.Dense(noOutput,'tanh', name='MLP_layer'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=customLRSchedule(hyperparams['initialLearningRate'], hyperparams['learningRateDecay'], hyperparams['decayDurationFactor'], hyperparams['numTrainingSteps']), \
                                        beta_1=hyperparams['beta1'], beta_2=hyperparams['beta2'], epsilon=hyperparams['epsilon'], amsgrad=False, name="tunedAdam_rnn")
    model.compile(optimizer=optimizer, loss = 'mse', metrics=[CustomMetricError(threshold=0.0)], run_eagerly=False)
    return model
    
def lstm_model(timestep, noInput, noOutput, batchSize):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(timestep, noInput),name='Input_layer'))
    model.add(tf.keras.layers.LSTM(units=noInput+noOutput, 
                   activation='tanh',
                   recurrent_activation='sigmoid',
                   unroll =False,
                   use_bias=True,
                   recurrent_dropout=0.0,
                   return_sequences=False))

    model.add(tf.keras.layers.Dense(noOutput,'tanh', name='MLP_layer'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=customLRSchedule(hyperparams['initialLearningRate'], hyperparams['learningRateDecay'], hyperparams['decayDurationFactor'], hyperparams['numTrainingSteps']), \
                                    beta_1=hyperparams['beta1'], beta_2=hyperparams['beta2'], epsilon=hyperparams['epsilon'], amsgrad=False, name="tunedAdam_lstm")
    model.compile(optimizer=optimizer, loss = 'mse', metrics=[CustomMetricError(threshold=0.0)], run_eagerly=False)
    return model
	
result_tf = {}
result_tf['lstm_lsr_wb_tf']= {}
for filename in fileslist:
    filepath = os.path.join(path,filename)
    result_tf['lstm_lsr_wb_tf'][filename] = []
    print('++', filename)
    timestep = int(filename.split('.')[4].split('s')[-1])
    with open(filepath, "r") as fp:
        [noIn, noOut] = [int(x) for x in fp.readline().replace('\n', '').split(',')]
    rdf = np.array(pd.read_csv(filepath, skiprows=1))
    df_val, df_train = train_test_split(rdf,test_size=hyperparams['testSize'])
    x_train, y_train = seperateValues(df_train, noIn, noOut)
    x_val, y_val = seperateValues(df_val, noIn, noOut)
    for i in range( x_train.shape[ 0 ] ) :
        for j in range( x_train.shape[ 1 ] ) :
            for k in range( x_train.shape[ 2 ] ) :
                x_train[ i, j, k ] = fromBit( x_train[ i, j, k ] )
    for i in range( y_train.shape[ 0 ] ) :
        for j in range( y_train.shape[ 1 ] ) :
            y_train[ i, j ] = fromBit( y_train[ i, j ] )
    for i in range( x_val.shape[ 0 ] ) :
        for j in range( x_val.shape[ 1 ] ) :
            for k in range( x_val.shape[ 2 ] ) :
                x_val[ i, j, k ] = fromBit( x_val[ i, j, k ] )
    for i in range( y_val.shape[ 0 ] ) :
        for j in range( y_val.shape[ 1 ] ) :
            y_val[ i, j ] = fromBit( y_val[ i, j ] )

    for run_count in range(5):
        model = lstm_model(timestep, noIn, noOut, int(hyperparams['batchSize']))
        model_history = model.fit(
                    x_train, # input
                    y_train, # output
                    batch_size=int(hyperparams['batchSize']),
                    verbose=1, # Suppress chatty output; use Tensorboard instead
                    epochs=int(hyperparams['numTrainingSteps']/(x_train.shape[0])),
                    validation_data=(x_val, y_val),
                    callbacks=[tf.keras.callbacks.EarlyStopping("val_custom_metric_error"), tf.keras.callbacks.TerminateOnNaN(), tf.keras.callbacks.TensorBoard(log_dir=logdir)]
                )
        y_pred = model.predict(x_val, verbose=0)
        result_tf['lstm_lsr_wb_tf'][filename].append(round(customMetricfn(y_val, y_pred), 5)*100)

results_dir_o = "./results/merge_results.csv"
for k, v in result_tf['lstm_lsr_wb_tf'].items():
    result_tf['lstm_lsr_wb_tf'][k] = round(sum(result_tf['lstm_lsr_wb_tf'][k]) / len(result_tf['lstm_lsr_wb_tf'][k]), 2)

lstm_lsr_wb_tf = pd.DataFrame.from_dict(result_tf['lstm_lsr_wb_tf'], orient='index', columns=['lstm_lsr_wb_tf'])

readfile = pd.read_csv(results_dir_o, index_col='dataset')
readfile.index.name = None
pd.concat([lstm_lsr_wb_tf, readfile], axis=1).to_csv(results_dir_o, index=True, index_label='dataset', mode='w') 

