




static trainParams screenParams = {
  .batchSize = 0,
  .numTrainingSteps = 0,
  .t = 1.0e300,
  .learningRate = 1.0e300,


.beta1=0.918977,
.beta2=0.957401,
.epsilon=0.000088,
.decayDurationFactor=0.949521,
.initialLearningRate=0.007574,
.learningRateDecay=0.000408,


  .numNodes = 4, // This is actually ignored in the code.

  .glorotScaleFactor=0.1,
  .orthogonalScaleFactor=0.1
  };


static trainParams valParams = {
  .batchSize = 0,
  .numTrainingSteps = 0,
  .t = 1.0e300,
  .learningRate = 1.0e300,

.beta1=0.982026,
.beta2=0.992750,
.epsilon=0.000155,
.decayDurationFactor=0.903809,
.initialLearningRate=0.009494,
.learningRateDecay=0.000022,


  .numNodes = 64, // This is actually ignored in the code.

  .glorotScaleFactor=0.1,
  .orthogonalScaleFactor=0.1
  };

static trainParams extParams = {
  .batchSize = 0,
  .numTrainingSteps = 0,
  .t = 1.0e300,
  .learningRate = 1.0e300,


.beta1=0.908417,
.beta2=0.998399,
.epsilon=0.004536,
.decayDurationFactor=0.951828,
.initialLearningRate=0.002694,
.learningRateDecay=0.000284,


  .numNodes = 64, // This is actually ignored in the code.

  .glorotScaleFactor=0.1,
  .orthogonalScaleFactor=0.1
  };


