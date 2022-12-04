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

import tensorboard
import keras
from keras.utils import tf_utils
import pandas as pd #pd.plotting.register_matplotlib_converters
import numpy as np
import sys, os, math, time, datetime, re

print("tf: ", tf.__version__)
print("tb: ", tensorboard.__version__)
print(os.getcwd())

DTYPE = tf.float64
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(1)
tf.config.set_visible_devices([], 'GPU')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

snapshot = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Debugging with Tensorboard
logdir="logs/fit/gru/" + snapshot
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

ISMOORE_DATASETS = False
FILESNUMBER = 1
LSTMNUMBER  = 1

with open("../../params/8sets_params/params_3W_1.txt") as f:
    hyperparams = dict([re.sub('['+' ,\n'+']','',x.replace(' .', '')).split('=') for x in f][1:-1])
hyperparams = dict([k, float(v)] for k, v in hyperparams.items())
hyperparams['testSize'] = 0.500
# hyperparams['noUnits'] = 81
print(hyperparams)

def seperateValues(data, noInput, noOutput, isMoore=True):
    x_data, y_data = None, None
    for i in range(data.shape[0]):
        if isMoore:
            x_data_i = data[i].reshape(-1, noInput+noOutput)
            x_data_i, y_data_i = x_data_i[:, 0:noInput], x_data_i[-1, noInput:]
        else:
            x_data_i = data[i][:-noOutput].reshape(-1, noInput)
            y_data_i = data[i][-noOutput:].reshape(-1, noOutput)
        x_data = x_data_i[np.newaxis,:,:] if x_data is None else np.append(x_data, x_data_i[np.newaxis,:,:], axis=0)
        y_data = y_data_i.reshape(1, -1)  if y_data is None else np.append(y_data, y_data_i.reshape(1, -1), axis=0)
    return x_data, y_data

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

def lstm_wLRS_wtCMF_model(noInput, noOutput, timestep):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(units=hyperparams['noUnits'], input_shape=[timestep, noInput],
                   activation='tanh', recurrent_activation='sigmoid', unroll=False, use_bias=True,
                   recurrent_dropout=0.0, return_sequences=False, name='LSTM_layer'))
    model.add(tf.keras.layers.Dense(noInput+noOutput, activation='tanh', name='MLP_layer'))
    model.add(tf.keras.layers.Dense(noOutput))
    optimizer = tf.keras.optimizers.Adam(learning_rate=customLRSchedule(hyperparams['batchSize'], hyperparams['initialLearningRate'], hyperparams['learningRateDecay'], hyperparams['decayDurationFactor'], hyperparams['numTrainingSteps']), \
                                    beta_1=hyperparams['beta1'], beta_2=hyperparams['beta2'], epsilon=hyperparams['epsilon'], amsgrad=False, name="tunedAdam_lstm")
    model.compile(optimizer=optimizer, loss = 'mse', run_eagerly=False)
    return model

def gru_wLRS_wtCMF_model(noInput, noOutput, timestep):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.GRU(units=int(hyperparams['noUnits']), input_shape=[timestep, noInput],
                   activation='tanh', recurrent_activation='sigmoid', unroll=False, use_bias=True,
                   recurrent_dropout=0.0, return_sequences=False, name='GRU_layer'))
    model.add(tf.keras.layers.Dense(noInput+noOutput, activation='tanh', name='MLP_layer'))
    model.add(tf.keras.layers.Dense(noOutput))
    optimizer = tf.keras.optimizers.Adam(learning_rate=customLRSchedule(hyperparams['batchSize'], hyperparams['initialLearningRate'], hyperparams['learningRateDecay'], hyperparams['decayDurationFactor'], hyperparams['numTrainingSteps']), \
                                    beta_1=hyperparams['beta1'], beta_2=hyperparams['beta2'], epsilon=hyperparams['epsilon'], amsgrad=False, name="tunedAdam_lstm")
    model.compile(optimizer=optimizer, loss = 'mse', run_eagerly=False)
    return model
    
if __name__ == '__main__':
    trainFile = f'3w_64.train'
    valFile = f'3w_64.val'
    testFile = f'3w_64.test'
    print(trainFile, valFile)
    df_train = np.array(pd.read_csv(os.path.join('.', trainFile ), skiprows=1))
    df_val = np.array(pd.read_csv(os.path.join('.', valFile ), skiprows=1))
    df_test = np.array(pd.read_csv(os.path.join('.', testFile ), skiprows=1))
    
    timestep = 64
    noIn, noOut = 8, 15
    print( "\nTrain and val shapes = ", df_train.shape, df_val.shape)
    
    print('Step 2: Separating values and labels.')
    x_train, y_train = seperateValues(df_train, noIn, noOut, isMoore=ISMOORE_DATASETS)
    x_val, y_val = seperateValues(df_val, noIn, noOut, isMoore=ISMOORE_DATASETS)
    x_test, y_test = seperateValues(df_test, noIn, noOut, isMoore=ISMOORE_DATASETS)
    scaler = StandardScaler()
    x_train = (scaler.fit_transform(x_train.reshape(x_train.shape[0], -1))).reshape(x_train.shape[0], timestep, noIn)
    x_val = (scaler.fit_transform(x_val.reshape(x_val.shape[0], -1))).reshape(x_val.shape[0], timestep, noIn)
    x_test = (scaler.fit_transform(x_test.reshape(x_val.shape[0], -1))).reshape(x_test.shape[0], timestep, noIn)

    print("+ Training set:   ", x_train.shape, y_train.shape, x_train.dtype)
    print("+ Validating set: ", x_val.shape, y_val.shape, x_val.dtype)
    print("+ Testing set: "   , x_test.shape, y_test.shape, x_test.dtype)
    
    metric =tf.keras.metrics.CategoricalAccuracy()
    print("----------------------- GRU TRAINING ----------------------")
    gru_model = gru_wLRS_wtCMF_model(noIn, noOut, timestep=timestep)
    model_history = gru_model.fit(
                        x_train, y_train,
                        batch_size=int(hyperparams['batchSize']),
                        verbose=1, # Suppress chatty output; use Tensorboard instead
                        # epochs=10,
                        epochs=int(hyperparams['numTrainingSteps']/(x_train.shape[0])),
                        validation_data=(x_val, y_val),
                        shuffle=True,
                        use_multiprocessing=False,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor=f"val_loss", patience=20, mode="min", start_from_epoch=20, restore_best_weights=True)]
                    )
    
    gru_val_y_pred      = gru_model.predict(x_val, verbose=1, batch_size=int(hyperparams['batchSize']))
    gru_val_performance = gru_model.evaluate(x_val, y_val, batch_size=int(hyperparams['batchSize']), verbose=1)
    print(f"GRU {valFile} val_performance = {gru_val_performance}")
    print(f"GRU {valFile} val_accuracy = {round(m.update_state(y_val, gru_val_y_pred).result().numpy(), 5)}")
    
    gru_test_y_pred      = gru_model.predict(x_test, verbose=1, batch_size=int(hyperparams['batchSize']))
    gru_test_performance = gru_model.evaluate(x_test, y_test, batch_size=int(hyperparams['batchSize']), verbose=1)
    print(f"GRU {testFile} test_performance = {gru_test_performance}")
    print(f"GRU {testFile} val_accuracy = {round(m.update_state(y_test, gru_test_y_pred).result().numpy(), 5)}")
    
    print("----------------------- LSTM TRAINING ----------------------")
    lstm_model = lstm_wLRS_wtCMF_model(noIn, noOut, timestep=timestep)
    model_history = lstm_model.fit(
                        x_train, y_train,
                        batch_size=int(hyperparams['batchSize']),
                        verbose=1, # Suppress chatty output; use Tensorboard instead
                        # epochs=10,
                        epochs=int(hyperparams['numTrainingSteps']/(x_train.shape[0])),
                        validation_data=(x_val, y_val),
                        shuffle=True,
                        use_multiprocessing=False,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor=f"val_loss", patience=20, mode="min", start_from_epoch=20, restore_best_weights=True)]
                    )
    lstm_val_y_pred      = lstm_model.predict(x_val, verbose=1, batch_size=int(hyperparams['batchSize']))
    lstm_val_performance = lstm_model.evaluate(x_val, y_val, batch_size=int(hyperparams['batchSize']), verbose=1)
    print(f"{valFile} val_performance = {lstm_val_performance}")
    print(f"{valFile} val_accuracy = {round(m.update_state(y_val, lstm_val_y_pred).result().numpy(), 5)}")
    
    lstm_test_y_pred      = lstm_model.predict(x_test, verbose=1, batch_size=int(hyperparams['batchSize']))
    lstm_test_performance = lstm_model.evaluate(x_test, y_test, batch_size=int(hyperparams['batchSize']), verbose=1)
    print(f"{testFile} test_performance = {lstm_test_performance}")
    print(f"{testFile} val_accuracy = {round(m.update_state(y_test, lstm_test_y_pred).result().numpy(), 5)}")