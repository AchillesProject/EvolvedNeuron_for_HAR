
static trainParams screenParams = {
  .batchSize = 0,
  .numTrainingSteps = 0,
  .t = 1.0e300,
  .learningRate = 1.0e300,


.beta1=0.998073,
.beta2=0.996617,
.epsilon=0.000150,
.decayDurationFactor=0.910931,
.initialLearningRate=0.000013,
.learningRateDecay=0.005511,

  .numNodes = 4, // This is actually ignored in the code.

  .glorotScaleFactor=0.1,
  .orthogonalScaleFactor=0.1
  };


static trainParams valParams = {
  .batchSize = 0,
  .numTrainingSteps = 0,
  .t = 1.0e300,
  .learningRate = 1.0e300,


.beta1=0.997825,
.beta2=0.968276,
.epsilon=0.000011,
.decayDurationFactor=0.946149,
.initialLearningRate=0.002108,
.learningRateDecay=0.001972,

  .numNodes = 64, // This is actually ignored in the code.

  .glorotScaleFactor=0.1,
  .orthogonalScaleFactor=0.1
  };

static trainParams extParams = {
  .batchSize = 0,
  .numTrainingSteps = 0,
  .t = 1.0e300,
  .learningRate = 1.0e300,


.beta1=0.983999,
.beta2=0.992698,
.epsilon=0.004096,
.decayDurationFactor=0.913129,
.initialLearningRate=0.004706,
.learningRateDecay=0.001776,


  .numNodes = 64, // This is actually ignored in the code.

  .glorotScaleFactor=0.1,
  .orthogonalScaleFactor=0.1
  };


