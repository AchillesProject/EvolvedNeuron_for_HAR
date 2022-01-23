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

def main(datasetpath, lossmethod):
    val_performance, train_time, training_history = {}, {}, {}
    # Getting data from csv file
    filepath = os.path.join(datasetpath)
    ni = datasetpath.split('.')[4].split('=')[1]
    no = datasetpath.split('.')[5].split('=')[1]
    mc = datasetpath.split('.')[6].split('=')[1]
    timestep = datasetpath.split('.')[7].split('s')[1]
    version = datasetpath.split('.')[8].split('n')[1]
    dataset  = datasetpath.split('.')[-2]
    filename = "{}_{}_{}_{}_{}_{}_{}".format(lossmethod[0], ni, no, mc, timestep, version, dataset)

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
    
    print('Step 3: Tuning....')
    log_dir_hparams = "../logs//hparams//" + filename + "//" + lossmethod[0] + "//"
    log_dir_tuner   = "../logs//tuner//"   + filename + "//" + lossmethod[0] + "//"
    lstm_tuner_v1=kt.tuners.Hyperband(
        tunner_lstm_model_v1,
        objective=kt.Objective('val_customMetricfn', direction='max'),
        max_epochs=8,
        seed=42,
        factor=3,
        hyperband_iterations=2,
        directory=log_dir_tuner,
        project_name="MasterProject",
        overwrite=True,
        logger=TensorBoardLogger(
            metrics=['loss','val_loss', 'customMetricfn', 'val_customMetricfn'], logdir=log_dir_hparams,
        ) # add only this argument
    )
    
    setup_tb(lstm_tuner_v1)  # (Optional) For more accurate visualization.
    lstm_tuner_v1.search(x_train, y_train,
                         epochs=8,
                         batch_size=1,
                         validation_data=(x_val, y_val),
                         shuffle=True,
                         use_multiprocessing=False,
#                          callbacks=[tf.keras.callbacks.TerminateOnNaN()]
                         callbacks=[tf.keras.callbacks.EarlyStopping("val_customMetricfn"), tf.keras.callbacks.TerminateOnNaN(), TerminateOnZero()]
                        )
    bestparams_v1 = lstm_tuner_v1.get_best_hyperparameters(1)[0]
    hyper_model_v1 = lstm_tuner_v1.hypermodel.build(bestparams_v1)
    
    print('Step 4: Rerun with the best tuning configuration.')
    strt_time = datetime.datetime.now() 
    training_history_v1 = hyper_model_v1.fit(
        x_train, y_train, 
        batch_size=1,
        verbose=1, # Suppress chatty output; use Tensorboard instead
        epochs=8,
        validation_data=(x_val, y_val),
        shuffle=True,
        callbacks=[tf.keras.callbacks.EarlyStopping("val_customMetricfn"), tf.keras.callbacks.TerminateOnNaN(), TerminateOnZero()]
    )
    curr_time = datetime.datetime.now()
    timedelta = curr_time - strt_time
    
    val_performance = hyper_model_v1.evaluate(x_val, y_val, batch_size=1, verbose=1)
    
    y_pred = hyper_model_v1.predict(x_val, verbose=1, batch_size=1)
    
    print('Step 5: Saving result.')
    tuning_result = {**{'project': lstm_tuner_v1.project_name},
                     **{'log_dir': lstm_tuner_v1.project_dir},
                     **{'dataset_no': filename},
                     **{'dataset_name': datasetpath},
                     **{'objectives': '{}, {}'.format(lstm_tuner_v1.oracle.objective.name, lstm_tuner_v1.oracle.objective.direction)},
                     **(lstm_tuner_v1.oracle.get_best_trials(1)[0].hyperparameters.values), 
                     **{'tuned_score': round(lstm_tuner_v1.oracle.get_best_trials(1)[0].score, 5)},
                     **{'loss': round(val_performance[0],5)},
                     **{'val_loss': round(val_performance[1],5)},
                     **{'val_customMetricfn': round(val_performance[2],5)},
                     **{'pred_accuracy': round(customMetricfn(y_val, y_pred), 5)},
                     **{'training_time (ms)': round(timedelta.total_seconds(),5)},
                    }
    [tuning_result.pop(key, None) for key in ['tuner/initial_epoch', 'tuner/bracket', 'tuner/round']]
    
    df = (pd.DataFrame.from_dict(tuning_result, orient='index', columns=[str(tuning_result['dataset_no'])]))
    df.to_csv('../results/{}.csv'.format(filename), index=True, index_label='Items')

if __name__=="__main__":
    print("No. of arguments passed is ", len(sys.argv))
    for idx, arg in enumerate(sys.argv):
        print("Argument #{} is {}".format(idx, arg))
    if len(sys.argv) == 3:
        lossmethod.pop()
        lossmethod.append(sys.argv[1])
        datasetpath = sys.argv[2]
    else:
        print("Don't have sufficient arguments.")
    
    main(datasetpath, lossmethod)
    
    print("Complete ", datasetpath)