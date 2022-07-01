


static trainParams screenParams = {
  .batchSize = 0,
  .numTrainingSteps = 0,
  .t = 1.0e300,
  .learningRate = 1.0e300,


.beta1=0.926605,
.beta2=0.999998,
.epsilon=0.000011,
.decayDurationFactor=0.925142,
.initialLearningRate=0.006633,
.learningRateDecay=0.000051,

  .numNodes = 4, // This is actually ignored in the code.

  .glorotScaleFactor=0.1,
  .orthogonalScaleFactor=0.1
  };


static trainParams valParams = {
  .batchSize = 0,
  .numTrainingSteps = 0,
  .t = 1.0e300,
  .learningRate = 1.0e300,

.beta1=0.998929,
.beta2=0.999969,
.epsilon=0.000181,
.decayDurationFactor=0.997863,
.initialLearningRate=0.001090,
.learningRateDecay=0.000290,

  .numNodes = 64, // This is actually ignored in the code.

  .glorotScaleFactor=0.1,
  .orthogonalScaleFactor=0.1
  };

static trainParams extParams = {
  .batchSize = 0,
  .numTrainingSteps = 0,
  .t = 1.0e300,
  .learningRate = 1.0e300,

.beta1=0.998072,
.beta2=0.998611,
.epsilon=0.000484,
.decayDurationFactor=0.912569,
.initialLearningRate=0.000412,
.learningRateDecay=0.000109,


  .numNodes = 64, // This is actually ignored in the code.

  .glorotScaleFactor=0.1,
  .orthogonalScaleFactor=0.1
  };


