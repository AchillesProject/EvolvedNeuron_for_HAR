

static trainParams screenParams = {
  .batchSize = 0,
  .numTrainingSteps = 0,
  .t = 1.0e300,
  .learningRate = 1.0e300,

.beta1=0.951986,
.beta2=0.999945,
.epsilon=0.000020,
.decayDurationFactor=0.957060,
.initialLearningRate=0.009615,
.learningRateDecay=0.001654,


  .numNodes = 4, // This is actually ignored in the code.

  .glorotScaleFactor=0.1,
  .orthogonalScaleFactor=0.1
  };


static trainParams valParams = {
  .batchSize = 0,
  .numTrainingSteps = 0,
  .t = 1.0e300,
  .learningRate = 1.0e300,

.beta1=0.935631,
.beta2=0.980963,
.epsilon=0.000088,
.decayDurationFactor=0.995805,
.initialLearningRate=0.007628,
.learningRateDecay=0.000114,


  .numNodes = 64, // This is actually ignored in the code.

  .glorotScaleFactor=0.1,
  .orthogonalScaleFactor=0.1
  };

static trainParams extParams = {
  .batchSize = 0,
  .numTrainingSteps = 0,
  .t = 1.0e300,
  .learningRate = 1.0e300,


.beta1=0.944793,
.beta2=0.999997,
.epsilon=0.000227,
.decayDurationFactor=0.974749,
.initialLearningRate=0.001167,
.learningRateDecay=0.000474,

  .numNodes = 64, // This is actually ignored in the code.

  .glorotScaleFactor=0.1,
  .orthogonalScaleFactor=0.1
  };


