import tensorflow as tf
import tensorboard
import pandas as pd
import numpy as np
import sys, os, math, time, datetime, re
from sklearn.model_selection import train_test_split

print("tf: ", tf.__version__)
print("tb: ", tensorboard.__version__)
print(os.getcwd())

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)
# tf.config.set_visible_devices([], 'GPU')
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
tf.keras.backend.set_floatx('float64')

# Debugging with Tensorboard
snapshot = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir="../logs/fit/rnn_v1_1/" + snapshot

with open("../params/params_5_bigsets.txt") as f:
    hyperparams = dict([re.sub('['+' ,\n'+']','',x.replace(' .', '')).split('=') for x in f][1:-1])
hyperparams = dict([k, float(v)] for k, v in hyperparams.items())
hyperparams['testSize'] = 0.5
print(hyperparams)

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
    return tf.keras.backend.mean(tf.equal(true, pred), axis=-1)

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
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.use_bias = True
    
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

class RNN_plus_models():
    def __init__(self, timestep, noInput, noOutput, batchSize, isLRS=False, isCMF=False):
        self.timestep = timestep
        self.noInput = noInput
        self.noOutput = noOutput
        self.batchSize = batchSize
        self.isLRS = isLRS
        self.isCMF = isCMF
    
    def rnn_plus_choose_models(self):
        if (self.isLRS and self.isCMF):
            return self.rnn_plus_wLRS_wCMF_model()
        elif (self.isLRS and not self.isCMF):
            return self.rnn_plus_wLRS_wtCMF_model()
        elif (not self.isLRS and self.isCMF):
            return self.rnn_plus_wtLRS_wCMF_model()
        else:
            return self.rnn_plus_wtLRS_wtCMF_model()
        
    def rnn_plus_wLRS_wCMF_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.RNN(cell=RNN_plus_v1_cell(units=self.noInput+self.noOutput), input_shape=[self.timestep, self.noInput], unroll=False, name='RNNp_layer'))
        model.add(tf.keras.layers.Dense(self.noOutput, activation='tanh', name='MLP_layer'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=customLRSchedule(hyperparams['initialLearningRate'], hyperparams['learningRateDecay'], hyperparams['decayDurationFactor'], hyperparams['numTrainingSteps']), \
                                            beta_1=hyperparams['beta1'], beta_2=hyperparams['beta2'], epsilon=hyperparams['epsilon'], amsgrad=False, name="tunedAdam")
        model.compile(optimizer=optimizer, loss = 'mse', metrics=[CustomMetricError(threshold=0.0)], run_eagerly=False)
        return model

    def rnn_plus_wLRS_wtCMF_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.RNN(cell=RNN_plus_v1_cell(units=self.noInput+self.noOutput), input_shape=[self.timestep, self.noInput], unroll=False, name='RNNp_layer'))
        model.add(tf.keras.layers.Dense(self.noOutput, activation='tanh', name='MLP_layer'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=customLRSchedule(hyperparams['initialLearningRate'], hyperparams['learningRateDecay'], hyperparams['decayDurationFactor'], hyperparams['numTrainingSteps']), \
                                            beta_1=hyperparams['beta1'], beta_2=hyperparams['beta2'], epsilon=hyperparams['epsilon'], amsgrad=False, name="tunedAdam")
        model.compile(optimizer=optimizer, loss = 'mse')
        return model

    def rnn_plus_wtLRS_wCMF_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.RNN(cell=RNN_plus_v1_cell(units=self.noInput+self.noOutput), input_shape=[self.timestep, self.noInput], unroll=False, name='RNNp_layer'))
        model.add(tf.keras.layers.Dense(self.noOutput, activation='tanh', name='MLP_layer'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0, name="Adam_wtlrs")
        model.compile(optimizer=optimizer, loss = 'mse', metrics=[CustomMetricError(threshold=0.0)], run_eagerly=False)
        return model

    def rnn_plus_wtLRS_wtCMF_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.RNN(cell=RNN_plus_v1_cell(units=self.noInput+self.noOutput), input_shape=[self.timestep, self.noInput], unroll=False, name='RNNp_layer'))
        model.add(tf.keras.layers.Dense(self.noOutput, activation='tanh', name='MLP_layer'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0, name="Adam_wtlrs")
        model.compile(optimizer=optimizer, loss = 'mse')
        return model
        
class RNN_models():
    def __init__(self, timestep, noInput, noOutput, batchSize, isLRS=False, isCMF=False):
        self.timestep = timestep
        self.noInput = noInput
        self.noOutput = noOutput
        self.batchSize = batchSize
        self.isLRS = isLRS
        self.isCMF = isCMF
        
    def rnn_choose_models(self):
        if (self.isLRS and self.isCMF):
            return self.rnn_wLRS_wCMF_model()
        elif (self.isLRS and not self.isCMF):
            return self.rnn_wLRS_wtCMF_model()
        elif (not self.isLRS and self.isCMF):
            return self.rnn_wtLRS_wCMF_model()
        else:
            return self.rnn_wtLRS_wtCMF_model()
        
    def rnn_wLRS_wCMF_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.RNN(cell=tf.keras.layers.SimpleRNNCell(units=self.noInput+self.noOutput), input_shape=[self.timestep, self.noInput], name='SimpleRNN_layer', stateful=False))
        model.add(tf.keras.layers.Dense(self.noOutput, activation='tanh', name='MLP_layer'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=customLRSchedule(hyperparams['initialLearningRate'], hyperparams['learningRateDecay'], hyperparams['decayDurationFactor'], hyperparams['numTrainingSteps']), \
                                            beta_1=hyperparams['beta1'], beta_2=hyperparams['beta2'], epsilon=hyperparams['epsilon'], amsgrad=False, name="tunedAdam_rnn")
        model.compile(optimizer=optimizer, loss = 'mse', metrics=[CustomMetricError(threshold=0.0)], run_eagerly=False)
        return model

    def rnn_wLRS_wtCMF_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.RNN(cell=tf.keras.layers.SimpleRNNCell(units=self.noInput+self.noOutput), input_shape=[self.timestep, self.noInput], name='SimpleRNN_layer', stateful=False))
        model.add(tf.keras.layers.Dense(self.noOutput, activation='tanh', name='MLP_layer'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=customLRSchedule(hyperparams['initialLearningRate'], hyperparams['learningRateDecay'], hyperparams['decayDurationFactor'], hyperparams['numTrainingSteps']), \
                                            beta_1=hyperparams['beta1'], beta_2=hyperparams['beta2'], epsilon=hyperparams['epsilon'], amsgrad=False, name="tunedAdam_rnn")
        model.compile(optimizer=optimizer, loss = 'mse')
        return model
    
    def rnn_wtLRS_wCMF_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.RNN(cell=tf.keras.layers.SimpleRNNCell(units=self.noInput+self.noOutput), input_shape=[self.timestep, self.noInput], name='SimpleRNN_layer', stateful=False))
        model.add(tf.keras.layers.Dense(self.noOutput, activation='tanh', name='MLP_layer'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0, name="Adam_wtlrs")
        model.compile(optimizer=optimizer, loss = 'mse', metrics=[CustomMetricError(threshold=0.0)], run_eagerly=False)
        return model
    
    def rnn_wtLRS_wtCMF_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.RNN(cell=tf.keras.layers.SimpleRNNCell(units=self.noInput+self.noOutput), input_shape=[self.timestep, self.noInput], name='SimpleRNN_layer', stateful=False))
        model.add(tf.keras.layers.Dense(self.noOutput, activation='tanh', name='MLP_layer'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0, name="Adam_wtlrs")
        model.compile(optimizer=optimizer, loss = 'mse')
        return model

class LSTM_models():
    def __init__(self, timestep, noInput, noOutput, batchSize, isLRS=False, isCMF=False):
        self.timestep = timestep
        self.noInput = noInput
        self.noOutput = noOutput
        self.batchSize = batchSize
        self.isLRS = isLRS
        self.isCMF = isCMF
        
    def lstm_choose_models(self):
        if (self.isLRS and self.isCMF):
            return self.lstm_wLRS_wCMF_model()
        elif (self.isLRS and not self.isCMF):
            return self.lstm_wLRS_wtCMF_model()
        elif (not self.isLRS and self.isCMF):
            return self.lstm_wtLRS_wCMF_model()
        else:
            return self.lstm_wtLRS_wtCMF_model()
        
    def lstm_wLRS_wCMF_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(units=self.noInput+self.noOutput, input_shape=[self.timestep, self.noInput],
                       activation='tanh', recurrent_activation='sigmoid', unroll =False, use_bias=True,
                       recurrent_dropout=0.0, return_sequences=False))
        model.add(tf.keras.layers.Dense(self.noOutput, activation='tanh', name='MLP_layer'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=customLRSchedule(hyperparams['initialLearningRate'], hyperparams['learningRateDecay'], hyperparams['decayDurationFactor'], hyperparams['numTrainingSteps']), \
                                        beta_1=hyperparams['beta1'], beta_2=hyperparams['beta2'], epsilon=hyperparams['epsilon'], amsgrad=False, name="tunedAdam_lstm")
        model.compile(optimizer=optimizer, loss = 'mse', metrics=[CustomMetricError(threshold=0.0)], run_eagerly=False)
        return model

    def lstm_wLRS_wtCMF_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(units=self.noInput+self.noOutput, input_shape=[self.timestep, self.noInput],
                       activation='tanh', recurrent_activation='sigmoid', unroll =False, use_bias=True,
                       recurrent_dropout=0.0, return_sequences=False))
        model.add(tf.keras.layers.Dense(self.noOutput, activation='tanh', name='MLP_layer'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=customLRSchedule(hyperparams['initialLearningRate'], hyperparams['learningRateDecay'], hyperparams['decayDurationFactor'], hyperparams['numTrainingSteps']), \
                                        beta_1=hyperparams['beta1'], beta_2=hyperparams['beta2'], epsilon=hyperparams['epsilon'], amsgrad=False, name="tunedAdam_lstm")
        model.compile(optimizer=optimizer, loss = 'mse')
        return model
    
    def lstm_wtLRS_wCMF_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(units=self.noInput+self.noOutput, input_shape=[self.timestep, self.noInput],
                       activation='tanh', recurrent_activation='sigmoid', unroll =False, use_bias=True, 
                       recurrent_dropout=0.0, return_sequences=False))
        model.add(tf.keras.layers.Dense(self.noOutput, activation='tanh', name='MLP_layer'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0, name="Adam_wtlrs")
        model.compile(optimizer=optimizer, loss = 'mse', metrics=[CustomMetricError(threshold=0.0)], run_eagerly=False)
        return model
    
    def lstm_wtLRS_wtCMF_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(units=self.noInput+self.noOutput, input_shape=[self.timestep, self.noInput],
                       activation='tanh', recurrent_activation='sigmoid', unroll =False, use_bias=True,
                       recurrent_dropout=0.0, return_sequences=False))
        model.add(tf.keras.layers.Dense(self.noOutput, activation='tanh', name='MLP_layer'))
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0, name="Adam_wtlrs")
        model.compile(optimizer=optimizer, loss = 'mse')
        return model

class models(RNN_models, RNN_plus_models, LSTM_models):
    def __init__(self,  timestep, noInput, noOutput, batchSize, modeltype='RNN_plus', isLRS=False, isCMF=False):
        super(models).__init__()
        self.modeltype = modeltype
        self.timestep = timestep
        self.noInput = noInput
        self.noOutput = noOutput
        self.batchSize = batchSize
        self.isLRS = isLRS
        self.isCMF = isCMF
    
    def chooseModels(self):
        sLRS = 'wLRS' if self.isLRS else 'wtLRS'
        sCMF = 'wCMF' if self.isCMF else 'wtCMF'
        prefix = f'{self.modeltype.lower()}_{sLRS}_{sCMF}_tf'
        print(f'Python script prefix: {prefix}')
        if self.modeltype == 'LSTM':
            return self.lstm_choose_models()
        elif self.modeltype == 'RNN_plus':
            return self.rnn_plus_choose_models()
        else:
            return self.rnn_choose_models()  

def main(datasetpath, modeltype, isLRS, isCMF):
    # Getting data from csv file
    filepath = os.path.join(datasetpath)
    file = filepath.split('/')[-1]
    ni = int(file.split('.')[1].split('=')[-1])
    no = int(file.split('.')[2].split('=')[-1])
    mc = int(file.split('.')[3].split('=')[-1])
    timestep = int(file.split('.')[4].split('s')[-1])
    sLRS = 'wLRS' if isLRS else 'wtLRS'
    sCMF = 'wCMF' if isCMF else 'wtCMF'
    prefix = f'{modeltype.lower()}_{sLRS}_{sCMF}_tf'
    filename = f"{prefix}_{ni}_{no}_{mc}_{timestep}"
    print('Dataset: ', filename)
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
    
    print('Step 3: Training....')
    model =  models(modeltype=modeltype, timestep=timestep, noInput=noIn, noOutput=noOut, batchSize=int(hyperparams['batchSize']), isLRS=isLRS, isCMF=isCMF).chooseModels()
    count = 0
    train_time, pred_performance = {}, {}
    for count in range(5): # Run n times
        strt_time = datetime.datetime.now() 
        model_history = model.fit(
                    x_train, y_train,
                    batch_size=int(hyperparams['batchSize']),
                    verbose=1, # Suppress chatty output; use Tensorboard instead
                    epochs=int(hyperparams['numTrainingSteps']/(x_train.shape[0])),
                    validation_data=(x_val, y_val),
                    shuffle=True,
                    use_multiprocessing=False
                )
        curr_time = datetime.datetime.now()
        train_time[count] = round((curr_time - strt_time).total_seconds(),5)
        pred_performance[count] = round(customMetricfn(y_val, model.predict(x_val, verbose=1, batch_size=int(hyperparams['batchSize']))), 5)*100
    
    print('Step 5: Saving result.')
    training_result = {**{'dataset_no': filename},
                       **{'dataset_name': datasetpath},
                       **{'traintime_avg (ms):': np.round(np.average(np.fromiter(train_time.values(), dtype=float)),decimals=5)},
                       **{'accuracy_avg': np.round(np.average(np.fromiter(pred_performance.values(), dtype=float)),decimals=5)},
                      }
    result_df = (pd.DataFrame.from_dict(training_result, orient='index', columns=[str(training_result['dataset_no'])]))
    result_df.to_csv('../results/{}.csv'.format(filename), index=True, index_label='Items')

if __name__=="__main__":
    print("No. of arguments passed is ", len(sys.argv))
    for idx, arg in enumerate(sys.argv):
        print("Argument #{} is {}".format(idx, arg))
    if len(sys.argv) == 5:
        datasetpath = str(sys.argv[1])
        modeltype   = str(sys.argv[2])
        isLRS       = bool(sys.argv[3])
        isCMF       = bool(sys.argv[4])
    else:
        print("Don't have sufficient arguments.")
    
    main(datasetpath, modeltype, isLRS, isCMF)
    
    print(f"Complete {datasetpath} with modeltype={modeltype}, isLRS={isLRS}, and isCMF={isCMF}")