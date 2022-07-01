
static trainParams screenParams = {
  .batchSize = 0,
  .numTrainingSteps = 0,
  .t = 1.0e300,
  .learningRate = 1.0e300,

.beta1=0.980035,
.beta2=0.996557,
.epsilon=0.000435,
.decayDurationFactor=0.941118,
.initialLearningRate=0.009221,
.learningRateDecay=0.000029,

  .numNodes = 4, // This is actually ignored in the code.

  .glorotScaleFactor=0.1,
  .orthogonalScaleFactor=0.1
  };


static trainParams valParams = {
  .batchSize = 0,
  .numTrainingSteps = 0,
  .t = 1.0e300,
  .learningRate = 1.0e300,

.beta1=0.993632,
.beta2=0.996080,
.epsilon=0.000038,
.decayDurationFactor=0.982885,
.initialLearningRate=0.007644,
.learningRateDecay=0.007822,

  .numNodes = 64, // This is actually ignored in the code.

  .glorotScaleFactor=0.1,
  .orthogonalScaleFactor=0.1
  };

static trainParams extParams = {
  .batchSize = 0,
  .numTrainingSteps = 0,
  .t = 1.0e300,
  .learningRate = 1.0e300,

.beta1=0.950198,
.beta2=0.996954,
.epsilon=0.003133,
.decayDurationFactor=0.921186,
.initialLearningRate=0.001856,
.learningRateDecay=0.000133,

  .numNodes = 64, // This is actually ignored in the code.

  .glorotScaleFactor=0.1,
  .orthogonalScaleFactor=0.1
  };












