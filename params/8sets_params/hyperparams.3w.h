

static trainParams screenParams = {
  .batchSize = 0,
  .numTrainingSteps = 0,
  .t = 1.0e300,
  .learningRate = 1.0e300,

.beta1=0.963139,
.beta2=0.999506,
.epsilon=0.000073,
.decayDurationFactor=0.982011,
.initialLearningRate=0.009740,
.learningRateDecay=0.000946,

  .numNodes = 4, // This is actually ignored in the code.

  .glorotScaleFactor=0.1,
  .orthogonalScaleFactor=0.1
  };


static trainParams valParams = {
  .batchSize = 0,
  .numTrainingSteps = 0,
  .t = 1.0e300,
  .learningRate = 1.0e300,

.beta1=0.959977,
.beta2=0.999954,
.epsilon=0.000805,
.decayDurationFactor=0.993567,
.initialLearningRate=0.007991,
.learningRateDecay=0.003591,


  .numNodes = 64, // This is actually ignored in the code.

  .glorotScaleFactor=0.1,
  .orthogonalScaleFactor=0.1
  };

static trainParams extParams = {
  .batchSize = 0,
  .numTrainingSteps = 0,
  .t = 1.0e300,
  .learningRate = 1.0e300,

.beta1=0.992102,
.beta2=0.999223,
.epsilon=0.000182,
.decayDurationFactor=0.905429,
.initialLearningRate=0.000677,
.learningRateDecay=0.002604,


  .numNodes = 64, // This is actually ignored in the code.

  .glorotScaleFactor=0.1,
  .orthogonalScaleFactor=0.1
  };


