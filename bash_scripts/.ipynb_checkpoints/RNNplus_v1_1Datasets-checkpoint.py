import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, LSTM, MaxPooling2D,AveragePooling2D,GlobalMaxPooling2D, GlobalAveragePooling2D, Flatten, Dropout, Reshape, BatchNormalization, ReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorboard
import keras_tuner as kt #(kt.tuners.RandomSearch, kt.tuners.Hyperband)
from kerastuner_tensorboard_logger import (
    TensorBoardLogger,
    setup_tb  # Optional
)
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, RobustScaler

from functools import partial
from matplotlib import rc, style
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
import pandas as pd #pd.plotting.register_matplotlib_converters
import numpy as np
from scipy import stats

import sys, os, math, time, datetime

print("kt: ", kt.__version__)
print("tf: ", tf.__version__)
print(os.getcwd())

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(1)

# disable GPU and anable MKL OneDNN
tf.config.set_visible_devices([], 'GPU')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['DNNL_VERBOSE'] = '0'

# reduce number of threads
os.environ['TF_NUM_INTEROP_THREADS'] = '1' 
os.environ['TF_NUM_INTRAOP_THREADS'] = '1' 

snapshot = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
path = '../Version9.128timesteps'
FILESNUMBER = 1
LSTMNUMBER  = 1

def fromBit( b ) :
    if b == 0.0 :
        return -0.9
    return 0.9
def isCorrect( target, actual ) :
    if target < 0.0 :
        y1 = False
    else :
        y1 = True
    if actual < 0.0 :
        y2 = False
    else :
        y2 = True
    return y1 == y2 
def customMetricfn(true, pred):
    count = 0
    numCorrect = 0
    for i in range( pred.shape[ 0 ] ) :
        for j in range( pred.shape[ 1 ] ) :
            count += 1
            if isCorrect( true[ i, j ], pred[ i, j ] ) :
                numCorrect += 1
    return (numCorrect/count)

class TerminateOnZero(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        value = logs.get('val_customMetricfn')
        if (value is not None):
            if value < 0.50:
                print(f'\nEpoch {epoch}: Val acccuracy {value} is too low, terminating training')
                self.model.stop_training = True
    
    def on_batch_end(self, batch, logs=None):
        value = logs.get('customMetricfn')
        if (value is not None):
            if value <= 0.40 and batch > 2000:
                print(f'\nBatch {batch}: Acccuracy {value} is too low, terminating training')
                self.model.stop_training = True

def seperateValues(data, noIn, noOut):
    x_data, y_data = None, None
    for i in range(data.shape[0]):
        x_data_i = data[i].reshape(-1, noIn+noOut).astype('float32')
        x_data_i, y_data_i = x_data_i[:, 0:noIn], x_data_i[-1, noIn:]
        x_data = x_data_i[np.newaxis,:,:] if x_data is None else np.append(x_data, x_data_i[np.newaxis,:,:], axis=0)
        y_data = y_data_i.reshape(1, -1) if y_data is None else np.append(y_data, y_data_i.reshape(1, -1), axis=0)
    return x_data, y_data
   
def srelu(x):
    return tf.keras.backend.clip(x, -1, 1)

class RNN_plus_v1_cell(tf.keras.layers.Layer):
    def __init__(self, units, kernel_initializer = 'glorot_uniform', recurrent_initializer = 'orthogonal', bias_initializer = 'zeros', **kwargs):
        self.units = units
        self.state_size = self.units
        self.output_size = self.units
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer
        super(RNN_plus_v1_cell, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units), name='w_i', initializer=self.kernel_initializer)
        self.recurrent_kernel = self.add_weight(shape=(self.units, self.units), name='w_o', initializer=self.recurrent_initializer)
        self.bias = self.add_weight( shape=(self.units,), name='b', initializer=self.bias_initializer)
        self.built = True
        
    def call(self, inputs, states, training=None):
        prev_output = states[0] if tf.nest.is_nested(states) else states
        i = tf.keras.backend.dot(inputs, self.kernel)
        z = tf.keras.backend.dot(prev_output, tf.linalg.set_diag(self.recurrent_kernel, np.zeros((self.units,), dtype=int)))
        v = (i + z)**2 - (i + z)
        output = srelu(v)

        new_state = [output] if tf.nest.is_nested(states) else output
        return output, new_state

def rnn_plus_v1_model():
    """Builds a recurrent model."""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(128, 3)))
    model.add(tf.keras.layers.RNN(cell=RNN_plus_v1_cell(units=8), unroll=True))
    model.add(tf.keras.layers.Dense(5,'tanh'))
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss = 'mse', metrics=['mse', customMetricfn], run_eagerly=False)
    return model
    

def main(datasetpath):
    # Getting data from csv file
    filepath = os.path.join(datasetpath)
    ni = datasetpath.split('.')[4].split('=')[1]
    no = datasetpath.split('.')[5].split('=')[1]
    mc = datasetpath.split('.')[6].split('=')[1]
    timestep = datasetpath.split('.')[7].split('s')[1]
    version = datasetpath.split('.')[8].split('n')[1]
    dataset  = datasetpath.split('.')[-2]
    filename = "{}_{}_{}_{}_{}_{}_{}".format('RNN_plus_v1', ni, no, mc, timestep, version, dataset)

    print('Dataset: ', filename)

    with open(filepath, "r") as fp:
        [noInput, noOutput] = [int(x) for x in fp.readline().split(',')]
    print("Number of Input and Output: ", noInput, noOutput)
    rdf = np.array(pd.read_csv(filepath, skiprows=1))
    
    print('Step 1: Dividing the training and testing set with ratio 1:1 (50%).')
    df_val, df_train = train_test_split(rdf,test_size=0.5)
    print(df_train.shape, df_val.shape)
    
    print('Step 2: Separating values and labels.')
    x_train, y_train = seperateValues(df_train, noInput, noOutput)
    x_val, y_val = seperateValues(df_val, noInput, noOutput)
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
    print("+ Training set:   ", x_train.shape, y_train.shape, x_train.dtype)
    print("+ Validating set: ", x_val.shape, y_val.shape, x_val.dtype)
    
    print('Step 3: Training....')
    rnn_model = rnn_plus_v1_model()
    count = 0
    val_performance, train_time, pred_performance = {}, {}, {}
    for count in range(3): # Run three times
        strt_time = datetime.datetime.now() 
        training_history_v1 = rnn_model.fit(
            x_train, y_train, 
            batch_size=1,
            verbose=1, # Suppress chatty output; use Tensorboard instead
            epochs=8,
            validation_data=(x_val, y_val),
            shuffle=True,
            callbacks=[tf.keras.callbacks.EarlyStopping("val_customMetricfn"), tf.keras.callbacks.TerminateOnNaN(), TerminateOnZero()]
        )
        curr_time = datetime.datetime.now()
        train_time[count] = round((curr_time - strt_time).total_seconds(),5)

        val_performance[count] = rnn_model.evaluate(x_val, y_val, batch_size=1, verbose=1)
        pred_performance[count] = round(customMetricfn(y_val, rnn_model.predict(x_val, verbose=1, batch_size=1)), 5)
    
    print('Step 5: Saving result.')
    training_result = {**{'dataset_no': filename},
                       **{'dataset_name': datasetpath},
                       **{'traintime_avg (ms):': np.round(np.average(np.fromiter(train_time.values(), dtype=float)),decimals=5)},
                       **{'rnnplus_avg': np.round(np.average(np.fromiter(pred_performance.values(), dtype=float)),decimals=5)},
                      }
    result_df = (pd.DataFrame.from_dict(training_result, orient='index', columns=[str(training_result['dataset_no'])]))
    result_df.to_csv('../results/{}.csv'.format(filename), index=True, index_label='Items')

if __name__=="__main__":
    print("No. of arguments passed is ", len(sys.argv))
    for idx, arg in enumerate(sys.argv):
        print("Argument #{} is {}".format(idx, arg))
    if len(sys.argv) == 2:
        datasetpath = sys.argv[1]
    else:
        print("Don't have sufficient arguments.")
    
    main(datasetpath)
    
>>>>>>> f449c52529ddd1fe388c35ad5f6075054aa9a0b6
    print("Complete ", datasetpath)
