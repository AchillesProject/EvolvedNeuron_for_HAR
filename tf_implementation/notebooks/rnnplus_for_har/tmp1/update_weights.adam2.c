/*
File: update_weights.adam2.c
Created: 2021-08-02
Modified: 2021-08-02

Changed to agree with the Adam in Tensorflow where epsilon actually is epsilon-hat
in the Kingma and Ba paper.

*/

#include <stdlib.h>
#include <math.h>
#include <cassert>
#include "matrix.h"
#include "allocate_nn.h"
#include "update_weights.h"

using namespace std;

void printTrainParams( trainParams params ) {
  printf( "  t = %lE  ", params.t );
  printf( "  learningRate = %lE\n\n", params.learningRate );
  printf( "  batchSize = %d\n", params.batchSize );
  printf( "  numTrainingSteps = %d\n", params.numTrainingSteps );

  printf( "  beta1  = %lE\n", params.beta1  );
  printf( "  beta2 = %lE\n", params.beta2 );
  printf( "  epsilon = %lE\n", params.epsilon );
  printf( "  decayDurationFactor  = %lE\n", params.decayDurationFactor  );
  printf( "  initialLearningRate  = %lE\n", params.initialLearningRate  );
  printf( "  learningRateDecay = %lE\n\n", params.learningRateDecay );

  printf( "  numNodes = %d\n\n", params.numNodes );

  printf( "  glorotScaleFactor = %lE\n", params.glorotScaleFactor );
  printf( "  orthogonalScaleFactor = %lE\n", params.orthogonalScaleFactor );

  fflush( stdout );
  return;
  }


static void updateVectorN( neuralNet& nn, aVector& w, const sargVector& sargW, aVector& m, aVector& v, trainParams params, int n ) {
  for( int i = 0; i < n; i++ ) {
    double g = nn.workSpace->diff( sargW[ i ] ) / (double) ( aadc::mmSize<mmType>() );
    m[i] = params.beta1 * m[i] + ( 1.0 - params.beta1 ) * g;
    v[i] = params.beta2 * v[i] + ( 1.0 - params.beta2 ) * g * g;
/*
    idouble mHat = 1.0 / ( 1.0 - pow( params.beta1, params.t ) ) * m[i];
    idouble vHat = 1.0 / ( 1.0 - pow( params.beta2, params.t ) ) * v[i];
    w[i] -= params.learningRate * mHat / ( sqrt( vHat ) + params.epsilon );
*/
    idouble alpha_t = 
      params.learningRate * sqrt( 1.0 - pow( params.beta2, params.t ) ) / ( 1.0 - pow( params.beta1, params.t ) );
    w[i] -= alpha_t  * m[ i ] / ( sqrt( v[ i ] ) + params.epsilon );


    }
  return;
  }

static void updateMatrix( neuralNet& nn, aMatrix& w, const sargMatrix& sargW, aMatrix& m, aMatrix& v, trainParams params ) {
  for( int i = 0; i < w.rows(); i++ ) 
    for( int k = 0; k < w.cols(); k++ ) {
      double g = nn.workSpace->diff( sargW[ i ][ k ] ) / (double) ( aadc::mmSize<mmType>() );
      m[i][ k ] = params.beta1 * m[i][ k ] + ( 1.0 - params.beta1 ) * g;
      v[i][ k ] = params.beta2 * v[i][ k ] + ( 1.0 - params.beta2 ) * g * g;
/*
      idouble mHat = 1.0 / ( 1.0 - pow( params.beta1, params.t ) ) * m[i][ k ];
      idouble vHat = 1.0 / ( 1.0 - pow( params.beta2, params.t ) ) * v[i][ k ];
      w[i][ k ] -= params.learningRate * mHat / ( sqrt( vHat ) + params.epsilon );
*/
    idouble alpha_t = 
      params.learningRate * sqrt( 1.0 - pow( params.beta2, params.t ) ) / ( 1.0 - pow( params.beta1, params.t ) );
    w[i][ k ] -= alpha_t  * m[ i ][ k ] / ( sqrt( v[ i ][ k ] ) + params.epsilon );

      }
  return;
  }

static void updateLayer( neuralNet& nn, layer& curr, trainParams params ) {
  if( curr.nt == DENSE ) {
//    printf( "\n\nupdateLayer: DENSE: updateMatrix  " );
    updateMatrix( nn, *curr.weightss[ 0 ], *curr.aWeightss[ 0 ], *curr.mWeightss[ 0 ], *curr.vWeightss[ 0 ], params );
//    printf( "\n\nupdateLayer: DENSE: updateVectorN  " );
    for( int i = 0; i < curr.numNodes; i++ )
      updateVectorN( nn, curr.auxWeights[ i ], curr.aAuxWeights[ i ], curr.mAuxWeights[ i ], curr.vAuxWeights[ i ], params, 1 );
    return;
    }
  if( curr.nt == LSTM ) {
//    printf( "\n\nupdateLayer: LSTM: updateMatrix 0  " );
    updateMatrix( nn, *curr.weightss[ 0 ], *curr.aWeightss[ 0 ], *curr.mWeightss[ 0 ], *curr.vWeightss[ 0 ], params );
//    printf( "\n\nupdateLayer: LSTM: updateMatrix 1  " );
    updateMatrix( nn, *curr.weightss[ 1 ], *curr.aWeightss[ 1 ], *curr.mWeightss[ 1 ], *curr.vWeightss[ 1 ], params );
//    printf( "\n\nupdateLayer: LSTM: updateMatrix 2  " );
    updateMatrix( nn, *curr.weightss[ 2 ], *curr.aWeightss[ 2 ], *curr.mWeightss[ 2 ], *curr.vWeightss[ 2 ], params );

//    printf( "\n\nupdateLayer: LSTM: updateVectorN  " );
    for( int i = 0; i < curr.numNodes; i++ )
      updateVectorN( nn, curr.auxWeights[ i ], curr.aAuxWeights[ i ], curr.mAuxWeights[ i ], curr.vAuxWeights[ i ], 
        params, numLstmAuxWeights );

//    printf( "\n\n\n" );
    return;

    }
  assert( false );
  }



/*
void updateWeights( neuralNet& nn, trainParams params ) {
  nn.numUpdates++;
  assert( nn.numLayers >= 2 );
  assert( nn.layers[ 0 ].nt == INPUT );

  params.learningRate = 0.001;
  params.beta1 = 0.9;
  params.beta2 = 0.999;
  params.epsilon = 1.0e-8;

  params.t = (double) nn.numUpdates;

  for( int li = 1; li < nn.numLayers; li++ )
    updateLayer( nn, nn.layers[ li ], params );
  return;
  }
*/


void updateWeights( neuralNet& nn, trainParams params ) {
  assert( nn.numLayers >= 2 );
  assert( nn.layers[ 0 ].nt == INPUT );

  nn.numUpdates++;
  params.t = (double) nn.numUpdates;

  double T = params.decayDurationFactor *  ( (double) params.numTrainingSteps ) / ( (double) params.batchSize );
  params.learningRate =
    params.t > T ?
      params.learningRateDecay * params.initialLearningRate
      :
      params.initialLearningRate - ( 1.0 - params.learningRateDecay ) * params.initialLearningRate * params.t / T;

  printf( "\nupdateWeights:\n" ); printTrainParams( params ); printf( "\n" );

  for( int li = 1; li < nn.numLayers; li++ )
    updateLayer( nn, nn.layers[ li ], params );
  return;
  }

