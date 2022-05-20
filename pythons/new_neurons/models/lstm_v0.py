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

RANDOM_SEED = 42
ISMOORE_DATASETS = True
timestep = 40
tf.random.set_seed(np.random.seed(RANDOM_SEED))
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(1)
tf.config.set_visible_devices([], 'GPU')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

# snapshot = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# path = '../../../../Datasets/6_har/0_WISDM/WISDM_ar_v1.1/WISDM_ar_v1.1_processed/WISDM_ar_v1.1_wt_overlap'
# Debugging with Tensorboard
# logdir="logs/fit/rnn_v1_1/" + snapshot
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
# tf.debugging.experimental.enable_dump_debug_info(logdir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)

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

class customLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, batchSize, initialLearningRate, learningRateDecay, decayDurationFactor, numTrainingSteps, glorotScaleFactor=0.1, orthogonalScaleFactor=0.1, name=None):
        self.batchSize = batchSize
        self.initialLearningRate = initialLearningRate
        self.learningRateDecay = learningRateDecay
        self.decayDurationFactor = decayDurationFactor
        self.glorotScaleFactor = glorotScaleFactor
        self.orthogonalScaleFactor = orthogonalScaleFactor
        self.numTrainingSteps = numTrainingSteps
        self.name = name
        self.T = tf.constant(self.decayDurationFactor * (self.numTrainingSteps/self.batchSize), dtype=tf.float32, name="T")
        self.lr = self.initialLearningRate
    
    def __call__(self, step):
        self.step = tf.cast(step, tf.float64)
        self.lr = tf.cond(self.step > self.T, 
                           lambda: tf.constant(self.learningRateDecay * self.initialLearningRate, dtype=tf.float64),
                           lambda: self.initialLearningRate * (1.0 - (1.0 - self.learningRateDecay) * self.step / self.T)
                          )
        return self.lr

def lstm_wLRS_wtCMF_model(noInput, noOutput, timestep):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=hyperparams['noUnits'], input_shape=[timestep, noInput],
                   activation='tanh', recurrent_activation='sigmoid', unroll=False, use_bias=True,
                   recurrent_dropout=0.0, return_sequences=False, name='LSTM_layer', dtype=tf.float64))
    model.add(tf.keras.layers.Dense(noInput+noOutput, activation='tanh', name='MLP_layer'))
    model.add(tf.keras.layers.Dense(noOutput))
    optimizer = tf.keras.optimizers.Adam(learning_rate=customLRSchedule(hyperparams['batchSize'], hyperparams['initialLearningRate'], hyperparams['learningRateDecay'], hyperparams['decayDurationFactor'], hyperparams['numTrainingSteps']), \
                                    beta_1=hyperparams['beta1'], beta_2=hyperparams['beta2'], epsilon=hyperparams['epsilon'], amsgrad=False, name="tunedAdam_lstm")
    model.compile(optimizer=optimizer, loss = 'mse', run_eagerly=False)
    return model


def indexOfMax( xs ) :
    m = xs[ 0 ]
    k = 0
    for i in range( 0, xs.size ) :
        if xs[ i ] > m :
            m = xs[ i ]
            k = i
    return k

#===============MAIN=================
if __name__ == '__main__':
    print('Step 1: Dividing the training and testing set with ratio 1:1 (50%).')
    ISMOORE_DATASETS = True
    noIn, noOut = 3, 6
    path = '../../Datasets/6_har/0_WISDM/WISDM_ar_v1.1/wisdm_script_and_data/wisdm_script_and_data/WISDM/testdata/' #fulla node1 path
    fileslist = [f for f in sorted(os.listdir(path)) if os.path.isfile(os.path.join(path, f))]
    
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

        model = lstm_wLRS_wtCMF_model(noIn, noOut, timestep=timestep)
        model_history = model.fit(
                            x_train, y_train,
                            batch_size=int(hyperparams['batchSize']),
                            verbose=1, # Suppress chatty output; use Tensorboard instead
                            # epochs=10,
                            epochs=int(hyperparams['numTrainingSteps']/(x_train.shape[0])),
                            validation_data=(x_val, y_val),
                            shuffle=True,
                            use_multiprocessing=False,
                            callbacks=[tensorboard_callback]
                        )
        y_pred = model.predict(x_val, verbose=1, batch_size=int(hyperparams['batchSize']))

        val_performance = model.evaluate(x_val, y_val, batch_size=int(hyperparams['batchSize']), verbose=1)
        print( f"\n{valFile}val_performance = {val_performance}\n")
        
        count = 0
        numCorrect = 0
        for i in range( y_pred.shape[ 0 ] ) :
            count += 1
            if indexOfMax( y_pred[ i ] ) == indexOfMax( y_val[ i ] ) :
                numCorrect += 1

        print( f"{valFile} val accuracy = {numCorrect / count}")
