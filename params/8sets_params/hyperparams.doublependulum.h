
static trainParams screenParams = {
  .batchSize = 0,
  .numTrainingSteps = 0,
  .t = 1.0e300,
  .learningRate = 1.0e300,


.beta1=0.917150,
.beta2=0.989469,
.epsilon=0.000017,
.decayDurationFactor=0.910262,
.initialLearningRate=0.008300,
.learningRateDecay=0.000039,

  .numNodes = 4, // This is actually ignored in the code.

  .glorotScaleFactor=0.1,
  .orthogonalScaleFactor=0.1
  };


static trainParams valParams = {
  .batchSize = 0,
  .numTrainingSteps = 0,
  .t = 1.0e300,
  .learningRate = 1.0e300,

.beta1=0.979515,
.beta2=0.999230,
.epsilon=0.001730,
.decayDurationFactor=0.962820,
.initialLearningRate=0.006841,
.learningRateDecay=0.000049,


  .numNodes = 64, // This is actually ignored in the code.

  .glorotScaleFactor=0.1,
  .orthogonalScaleFactor=0.1
  };

static trainParams extParams = {
  .batchSize = 0,
  .numTrainingSteps = 0,
  .t = 1.0e300,
  .learningRate = 1.0e300,


.beta1=0.981567,
.beta2=0.999981,
.epsilon=0.000722,
.decayDurationFactor=0.912592,
.initialLearningRate=0.004732,
.learningRateDecay=0.000012,


  .numNodes = 64, // This is actually ignored in the code.

  .glorotScaleFactor=0.1,
  .orthogonalScaleFactor=0.1
  };


