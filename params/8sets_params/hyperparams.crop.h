
static trainParams screenParams = {
  .batchSize = 0,
  .numTrainingSteps = 0,
  .t = 1.0e300,
  .learningRate = 1.0e300,


.beta1=0.964797,
.beta2=0.997529,
.epsilon=0.000015,
.decayDurationFactor=0.921812,
.initialLearningRate=0.006147,
.learningRateDecay=0.000139,

  .numNodes = 4, // This is actually ignored in the code.

  .glorotScaleFactor=0.1,
  .orthogonalScaleFactor=0.1
  };


static trainParams valParams = {
  .batchSize = 0,
  .numTrainingSteps = 0,
  .t = 1.0e300,
  .learningRate = 1.0e300,

.beta1=0.926130,
.beta2=0.999997,
.epsilon=0.000057,
.decayDurationFactor=0.953564,
.initialLearningRate=0.008128,
.learningRateDecay=0.000011,


  .numNodes = 64, // This is actually ignored in the code.

  .glorotScaleFactor=0.1,
  .orthogonalScaleFactor=0.1
  };

static trainParams extParams = {
  .batchSize = 0,
  .numTrainingSteps = 0,
  .t = 1.0e300,
  .learningRate = 1.0e300,


.beta1=0.928963,
.beta2=0.993591,
.epsilon=0.000022,
.decayDurationFactor=0.900179,
.initialLearningRate=0.002653,
.learningRateDecay=0.005072,


  .numNodes = 64, // This is actually ignored in the code.

  .glorotScaleFactor=0.1,
  .orthogonalScaleFactor=0.1
  };


