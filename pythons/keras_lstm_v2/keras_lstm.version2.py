import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, LSTM, MaxPooling2D,AveragePooling2D,GlobalMaxPooling2D, GlobalAveragePooling2D, Flatten, Dropout, Reshape, BatchNormalization, ReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorboard
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


# reduce number of threads
os.environ['TF_NUM_INTEROP_THREADS'] = '1' 
os.environ['TF_NUM_INTRAOP_THREADS'] = '1' 

#matplotlib inline
#config InlineBackend.figure_format='retina'

style.use("seaborn")
pd.plotting.register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale = 1)

# rcParams['figure.figsize'] = 22, 10

# RANDOM_SEED = 42

# np.random.seed(RANDOM_SEED)
# tf.random.set_seed(RANDOM_SEED)
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(1)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   try:
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#   except RuntimeError as e:
#     print(e)

# tf.debugging.set_log_device_placement(False)

# strategy = tf.distribute.MirroredStrategy()

# Writing to file
filepath = './Version9.128timesteps/seqnetdata.ni=3.no=5.mc=15.numTimeSteps128.version9.4.csv'
with open(filepath, "r") as fp:
    [noInput, noOutput] = [int(x) for x in fp.readline().split(',')]
rdf = np.array(pd.read_csv(filepath, skiprows=1))
print(type(rdf), rdf.shape)
print(type(noInput), noInput, type(noOutput), noOutput)
# display(rdf)
# np.set_printoptions(threshold=1000)
def shifting(bitlist):
    out = 0
    for bit in bitlist:
        out = (out << 1) | bit
    return out

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

# Validating set
x_test, y_test = None, None
for i in range(df_val.shape[0]):
    df_val_i = df_val[i].reshape(-1, noInput+noOutput).astype('float32')
    x_val_i, y_val_i = df_val_i[:, 0:noInput], df_val_i[-1, noInput:]
    x_val = x_val_i[np.newaxis,:,:] if x_val is None else np.append(x_val, x_val_i[np.newaxis,:,:], axis=0)
    y_val = y_val_i.reshape(1,-1) if y_val is None else np.append(y_val, y_val_i.reshape(1,-1), axis=0)
print("+ Validating set: ", x_val.shape, y_val.shape, x_val.dtype)

print('Step 3: Normalizing the labels.')
# Training set
y_train_norm = []
for i in range(y_train.shape[0]):
    y_train_norm.append(shifting(y_train[i,:].astype('int32')))
    
enc = OneHotEncoder(handle_unknown='ignore', sparse=False).fit(np.array(y_train_norm).reshape(-1, 1))
y_train_norm = enc.transform(np.array(y_train_norm).reshape(-1, 1))
print("+ Normalizied training set:   ", y_train_norm.shape)
# Validating set
y_val_norm = []
for i in range(y_val.shape[0]):
    y_val_norm.append(shifting(y_val[i,:].astype('int32')))
    
enc = OneHotEncoder(handle_unknown='ignore', sparse=False).fit(np.array(y_val_norm).reshape(-1, 1))
y_val_norm = enc.transform(np.array(y_val_norm).reshape(-1, 1))
print("+ Normalizied validating set: ", y_val_norm.shape)

# with strategy.scope(): 
lstm_model = tf.keras.Sequential()
lstm_model.add(
  tf.keras.layers.LSTM(
      units=8, 
      input_shape=[x_train.shape[1], x_train.shape[2]],
      activation='tanh', recurrent_activation='sigmoid',
      unroll =False,
      use_bias=True,
      recurrent_dropout=0,
      return_sequences=False
  )
)

lstm_model.add(tf.keras.layers.Dense(y_train.shape[1], activation='sigmoid'))
adam = Adam(learning_rate = 0.001, decay=0)
# lstm_model.compile(loss='mse', optimizer=adam, metrics='mse')
lstm_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)])

strt_time = datetime.datetime.now() 
lstm_model.fit(
    x_train, y_train,
    epochs = 8,
    batch_size=1,
    verbose=0,
    validation_data=(x_val, y_val),
    shuffle=True,
    use_multiprocessing=False
)
curr_time = datetime.datetime.now()
# display(lstm_model.summary())
timedelta = curr_time - strt_time
dnn_train_time = timedelta.total_seconds()

val_performance = lstm_model.evaluate(x_val, y_val)

y_pred = lstm_model.predict(x_val, verbose=0)
print(y_pred)

m = tf.keras.metrics.BinaryAccuracy()
m.update_state(y_pred, y_val)
print(m.result().numpy())

y_pred[np.where(y_pred < 0.5)] = 0
y_pred[np.where(y_pred >= 0.5)] = 1
# display(y_pred.astype('int32'))
# display(y_val.astype('int32'))
count = 0
# for i in range(y_pred.shape[0]):
#     if (np.array_equal(y_pred[i].astype('int32'), y_val[i].astype('int32'))):
#         count += 1
print( val_performance[1], val_performance[1]*y_pred.shape[0])
