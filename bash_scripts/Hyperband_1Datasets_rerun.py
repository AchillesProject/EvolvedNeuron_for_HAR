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
LOOP_NUMBER = 3

lstm_af = ['tanh', 'relu', 'sigmoid', 'softmax', 'softsign', 'selu', 'elu']
lstm_raf = ['sigmoid']
dense_af = ['tanh','sigmoid', 'softmax', 'softsign']
learning_rates = [1e-3, 1e-2, 1e-4]
thresholds = [0.5, 0.51, 0.6, 0.7]
optimizers = ['adam', 'sgd']
lossmethod = ['mse']

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
   
def tunner_lstm_model_v1(hp):
    """Builds a recurrent model."""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(128, 3)))
    model.add(tf.keras.layers.LSTM(units=8, 
                   activation=hp.Choice('af_LSTM', lstm_af),
                   recurrent_activation=hp.Choice('raf_LSTM', lstm_raf),
                   unroll =False,
                   use_bias=True,
                   recurrent_dropout=0,
                   return_sequences=False))

    model.add(tf.keras.layers.Dense(5, hp.Choice('af_dense', dense_af)))
    if (hp.Choice('optimizer', optimizers) == 'adam'):
        optimizer = tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=learning_rates))
    else:
        optimizer = tf.keras.optimizers.SGD(hp.Choice('learning_rate', values=learning_rates))

    model.compile(optimizer=optimizer, loss = 'mse', metrics=['mse', customMetricfn])

    return model

def main(resultpath, lossmethod):
    print(f'Starting with {resultpath}')
    result = pd.read_csv(resultpath, index_col=0)
    dataset_no = result.loc['dataset_no'].values[0].split('_')
    dataset_name = f'seqnetdata.ni={dataset_no[1]}.no={dataset_no[2]}.mc={dataset_no[3]}.numTimeSteps{dataset_no[4]}.version{dataset_no[5]}.{dataset_no[6]}.csv'
    af_LSTM = result.loc['af_LSTM'].values[0]
    raf_LSTM = result.loc['raf_LSTM'].values[0]
    af_dense = result.loc['af_dense'].values[0]
    optimizer_dense = result.loc['optimizer'].values[0]
    learningrate = float(result.loc['learning_rate'].values[0])
    epoch_no = int(result.loc['tuner/epochs'].values[0])

    datapath = f'../Version9.128timesteps/{dataset_name}'
    
    with open(datapath, "r") as fp:
        [noInput, noOutput] = [int(x) for x in fp.readline().split(',')]
    rdf = np.array(pd.read_csv(datapath, skiprows=1))

    print('Step 1: Dividing the training and testing set with ratio 1:1 (50%).')
    df_val, df_train = train_test_split(rdf,test_size=0.5)
    print(df_train.shape, df_val.shape)

    print('Step 2: Separating values and labels.')
    # Training set
    x_train, y_train, x_val, y_val = None, None, None, None
    for i in range(df_train.shape[0]):
        df_train_i = df_train[i].reshape(-1, noInput+noOutput).astype('float32')
        x_train_i, y_train_i = df_train_i[:, 0:noInput], df_train_i[-1, noInput:]
        x_train = x_train_i[np.newaxis,:,:] if x_train is None else np.append(x_train, x_train_i[np.newaxis,:,:], axis=0)
        y_train = y_train_i.reshape(1, -1) if y_train is None else np.append(y_train, y_train_i.reshape(1, -1), axis=0)
    print("+ Training set:   ", x_train.shape, y_train.shape, x_train.dtype)

    # Validation set
    x_test, y_test = None, None
    for i in range(df_val.shape[0]):
        df_val_i = df_val[i].reshape(-1, noInput+noOutput).astype('float32')
        x_val_i, y_val_i = df_val_i[:, 0:noInput], df_val_i[-1, noInput:]
        x_val = x_val_i[np.newaxis,:,:] if x_val is None else np.append(x_val, x_val_i[np.newaxis,:,:], axis=0)
        y_val = y_val_i.reshape(1,-1) if y_val is None else np.append(y_val, y_val_i.reshape(1,-1), axis=0)
    print("+ Validation set: ", x_val.shape, y_val.shape, x_val.dtype)

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

    lstm_model = tf.keras.Sequential()
    lstm_model.add(
      tf.keras.layers.LSTM(
          units=8, 
          input_shape=[x_train.shape[1], x_train.shape[2]],
          activation=af_LSTM, recurrent_activation=raf_LSTM,
          unroll =False,
          use_bias=True,
          recurrent_dropout=0,
          return_sequences=False
      )
    )

    lstm_model.add(tf.keras.layers.Dense(y_train.shape[1], activation=af_dense))
    if(optimizer_dense == 'adam'):
        lstm_model.compile(loss='mse', optimizer=Adam(learning_rate = learningrate, decay=0), metrics=['mse', customMetricfn])
    else:
        lstm_model.compile(loss='mse', optimizer=SGD(learning_rate = learningrate), metrics=['mse', customMetricfn])
    
    for loop in range(LOOP_NUMBER):
        strt_time = datetime.datetime.now() 
        
        lstm_model.fit(
            x_train, y_train, 
            batch_size=1,
            verbose=1, # Suppress chatty output; use Tensorboard instead
            epochs=epoch_no,
            validation_data=(x_val, y_val),
            shuffle=True,
            callbacks=[tf.keras.callbacks.EarlyStopping("val_customMetricfn"), tf.keras.callbacks.TerminateOnNaN(), TerminateOnZero()],
        )
        
        curr_time = datetime.datetime.now()
        timedelta = curr_time - strt_time
        dnn_train_time = timedelta.total_seconds()
        
        val_performance = lstm_model.evaluate(x_val, y_val, batch_size=1, verbose=1)

        y_pred = lstm_model.predict(x_val, verbose=1, batch_size=1)

        result.loc[f'rerun_acc_{loop}'] = round(customMetricfn(y_val, y_pred), 5)
    result.to_csv(resultpath, index=True)

if __name__=="__main__":
    print("No. of arguments passed is ", len(sys.argv))
    for idx, arg in enumerate(sys.argv):
        print("Argument #{} is {}".format(idx, arg))
    if len(sys.argv) == 3:
        lossmethod.pop()
        lossmethod.append(sys.argv[1])
        resultpath = sys.argv[2]
    else:
        print("Don't have sufficient arguments.")
    
    main(resultpath, lossmethod)
    
    print("Complete ", resultpath)