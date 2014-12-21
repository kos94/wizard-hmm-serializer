package com.wizardfight.recognition;

import java.io.Serializable;

/**
 * This class implements a discrete Hidden Markov Model.
 */
class HiddenMarkovModel implements Serializable {

    private static final long serialVersionUID = 1L;
    int numStates = 0; // The number of states for this model
    int[] estimatedStates = new int[0];

    double[] pi; // The state start probability vector
   
    double[][] a; // The transitions probability matrix
    double[][] b; // The emissions probability matrix

    double predict(int[] obs) {
        final int N = numStates;
        final int T = obs.length;
        int t, i, j;
        MatrixDouble alpha = new MatrixDouble(T, numStates);
        double[] c = new double[T];

	// //////////////// Run the forward algorithm ////////////////////////
        // Step 1: Init at t=0
        t = 0;
        c[t] = 0.0;
        for (i = 0; i < N; i++) {
            double val = pi[i] * b[i][obs[t]];
            alpha.dataPtr[t][i] = val;
            c[t] += val;
        }

        // Set the inital scaling coeff
        c[t] = 1.0 / c[t];

        // Scale alpha
        for (i = 0; i < N; i++) {
            double val = alpha.dataPtr[t][i];
            val *= c[t];
            alpha.dataPtr[t][i] = val;
        }

        // Step 2: Induction
        for (t = 1; t < T; t++) {
            c[t] = 0.0;
            for (j = 0; j < N; j++) {
                alpha.dataPtr[t][j] = 0.0;
                for (i = 0; i < N; i++) {
                    double val = alpha.dataPtr[t][j];
                    val += alpha.dataPtr[t - 1][i] * a[i][j];
                    alpha.dataPtr[t][j] = val;

                }
                double val = alpha.dataPtr[t][j];
                val *= b[j][obs[t]];
                alpha.dataPtr[t][j] = val;
                c[t] += alpha.dataPtr[t][j];
            }

            // Set the scaling coeff
            c[t] = 1.0 / c[t];

            // Scale Alpha
            for (j = 0; j < N; j++) {
                double val = alpha.dataPtr[t][j];
                val *= c[t];
                alpha.dataPtr[t][j] = val;
            }
        }

        if (estimatedStates.length != T) {
            estimatedStates = new int[T];
        }
        for (t = 0; t < T; t++) {
            double maxValue = 0;
            for (i = 0; i < N; i++) {
                if (alpha.dataPtr[t][i] > maxValue) {
                    maxValue = alpha.dataPtr[t][i];
                    estimatedStates[t] = i;
                }
            }
        }

        // Termination
        double loglikelihood = 0.0;
        for (t = 0; t < T; t++) {
            loglikelihood += Math.log(c[t]);
        }
        return -loglikelihood; // Return the negative log likelihood
    }
    
     public void print() {
//        System.out.println(trained + " " + useScaling + " " + useNullRejection);
//        System.out.println(delta + " " + maxNumIter + " " + numClasses);
//        System.out.println(numInputDimensions + " " + numRandomTrainingIterations + " " + numStates);
//        System.out.println(numSymbols + " " + predictedClassLabel);
//        System.out.println(bestDistance + " " + maxLikelihood + " " + minImprovement);
//        for(int i=0; i<classLabels.length; i++) {
//            System.out.print(classLabels[i] + " ");
//        }
//        System.out.println();
//        
//        for(int i=0; i<classLikelihoods.length; i++) {
//            System.out.print(classLikelihoods[i] + " ");
//        }
//        System.out.println();
//        
//        for(int i=0; i<classDistances.length; i++) {
//            System.out.print(classDistances[i] + " ");
//        }
//        System.out.println();
//        
//        for(int i=0; i<nullRejectionThresholds.length; i++) {
//            System.out.print(nullRejectionThresholds[i] + " ");
//        }
//        System.out.println();
//        
//        System.out.println(modelType);
//        
//        for(HiddenMarkovModel hmm : models) {
//            hmm.print();
//        }
        System.out.println("\n==========================");
    }
    
     public void printTXT() {
        System.out.println("NumStates: " + numStates);
        System.out.println("A: ");
        for(int i=0; i<a.length; i++) {
            for(int j=0; j<a[0].length; j++) {
                System.out.print(a[i][j] + " ");
            }
            System.out.println();
        }
        System.out.println("B: ");
        for(int i=0; i<b.length; i++) {
            for(int j=0; j<b[0].length; j++) {
                System.out.print(b[i][j] + " ");
            }
            System.out.println();
        }
        System.out.println("Pi: ");
        for(int i=0; i<pi.length; i++) {
            System.out.print(pi[i]);
            if(i != pi.length - 1) System.out.print(" ");
            else System.out.println();
        }
    }
}
