
static trainParams screenParams = {
  .batchSize = 0,
  .numTrainingSteps = 0,
  .t = 1.0e300,
  .learningRate = 1.0e300,

.beta1=0.941672,
.beta2=0.990943,
.epsilon=0.000017,
.decayDurationFactor=0.943000,
.initialLearningRate=0.007922,
.learningRateDecay=0.004858,


  .numNodes = 4, // This is actually ignored in the code.

  .glorotScaleFactor=0.1,
  .orthogonalScaleFactor=0.1
  };


static trainParams valParams = {
  .batchSize = 0,
  .numTrainingSteps = 0,
  .t = 1.0e300,
  .learningRate = 1.0e300,

.beta1=0.983699,
.beta2=0.963457,
.epsilon=0.001990,
.decayDurationFactor=0.900493,
.initialLearningRate=0.009561,
.learningRateDecay=0.000019,


  .numNodes = 64, // This is actually ignored in the code.

  .glorotScaleFactor=0.1,
  .orthogonalScaleFactor=0.1
  };

static trainParams extParams = {
  .batchSize = 0,
  .numTrainingSteps = 0,
  .t = 1.0e300,
  .learningRate = 1.0e300,


.beta1=0.972003,
.beta2=0.936829,
.epsilon=0.000242,
.decayDurationFactor=0.982879,
.initialLearningRate=0.007921,
.learningRateDecay=0.000022,


  .numNodes = 64, // This is actually ignored in the code.

  .glorotScaleFactor=0.1,
  .orthogonalScaleFactor=0.1
  };


