package com.wizardfight.recognition;

import java.util.ArrayList;
import java.io.Serializable;

/**
 * This class implements a discrete Hidden Markov Model.
 */
public class HiddenMarkovModel implements Serializable {

    private static final long serialVersionUID = 1L;

    boolean modelTrained = false;

    int numStates = 0; // The number of states for this model
    int numSymbols = 0; // The number of symbols for this model
    int delta = 1; // The number of states a model can move to in a
    // HMMModelTipes.LEFTRIGHT model
    int numRandomTrainingIterations = 5; // The number of training loops to find
    // the best starting values
    int maxNumIter = 100; // The maximum number of iter allowed during the full
    // training

    double logLikelihood = 0.0; // The log likelihood of an observation sequence
    // given the modal, calculated by the forward
    // method
    double cThreshold = -1000; // The classification threshold for this model
    double minImprovement = 1.0e-5; // The minimum improvement value for the
    // training loop

    int[] observationSequence = new int[0];
    int[] estimatedStates = new int[0];

    double[] pi; // The state start probability vector

    HMMModelTypes modelType = HMMModelTypes.ERGODIC;

    ArrayList<Double> trainingIterationLog = new ArrayList<Double>(); 
    // Stores the loglikelihood at each iteration the BaumWelch algorithm

    MatrixDouble a = new MatrixDouble(); // The transitions probability matrix
    MatrixDouble b = new MatrixDouble(); // The emissions probability matrix

    private double getRandomNumberUniform(double minRange, double maxRange) {
        return (Math.random() * (maxRange - minRange)) + minRange;
    }
    
    double predict(int[] obs) {
        final int N = numStates;
        final int T = obs.length;
        int t, i, j = 0;
        MatrixDouble alpha = new MatrixDouble(T, numStates);
        double[] c = new double[T];

	// //////////////// Run the forward algorithm ////////////////////////
        // Step 1: Init at t=0
        t = 0;
        c[t] = 0.0;
        for (i = 0; i < N; i++) {
            double val = pi[i] * b.dataPtr[i][obs[t]];
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
                    val += alpha.dataPtr[t - 1][i] * a.dataPtr[i][j];
                    alpha.dataPtr[t][j] = val;

                }
                double val = alpha.dataPtr[t][j];
                val *= b.dataPtr[j][obs[t]];
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
        System.out.println("modelTrained: " + modelTrained + " " + numStates + " " + numSymbols);
        System.out.println(delta + " " + numRandomTrainingIterations + " " + maxNumIter);
        System.out.println(logLikelihood + " " + cThreshold + " " + minImprovement);
        for(int i=0; i<observationSequence.length; i++) {
            System.out.print(observationSequence[i] + " ");
        }
        System.out.println();
        
        for(int i=0; i<estimatedStates.length; i++) {
            System.out.print(estimatedStates[i] + " ");
        }
        System.out.println();
        
        for(int i=0; i<pi.length; i++) {
            System.out.print(pi[i] + " ");
        }
        System.out.println();
        
        System.out.println(modelType);
        
        for(Double d: trainingIterationLog) {
            System.out.print(d + " ");
        }
        System.out.println();
        a.print();
        b.print();
    }
    
    public void printTXT() {
        System.out.println("NumStates: " + numStates);
        System.out.println("NumSymbols: " + numSymbols);
        System.out.println("ModelType: " + ((modelType == HMMModelTypes.ERGODIC)? 0 : 1));
        System.out.println("Delta: " + delta);
        System.out.println("Threshold: " + cThreshold);
        System.out.println("NumRandomTrainingIterations: " + numRandomTrainingIterations);
        System.out.println("MaxNumIter: " + maxNumIter);
        System.out.println("A: ");
        a.print();
        System.out.println("B: ");
        b.print();
        System.out.println("Pi: ");
        for(int i=0; i<pi.length; i++) {
            System.out.print(pi[i]);
            if(i != pi.length - 1) System.out.print(" ");
            else System.out.println();
        }
    }
}
