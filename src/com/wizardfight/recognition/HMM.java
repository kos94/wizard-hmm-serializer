package com.wizardfight.recognition;
import java.io.Serializable;
import java.util.ArrayList;


/**
 * This class acts as the main interface for using a Hidden Markov Model.
 */
public class HMM implements Serializable {

    private static final long serialVersionUID = 2L;
    // Variables for all the HMMs
    boolean useNullRejection = false; // would be useful if failed gesture detection

     int numClasses;
    private int predictedClassLabel;

    double bestDistance; 
    double maxLikelihood; 
    // for each model during training
    int[] classLabels = new int[0];
    double[] classLikelihoods = new double[0];
    double[] classDistances = new double[0];
    double[] nullRejectionThresholds;


    final ArrayList<HiddenMarkovModel> models = new ArrayList<HiddenMarkovModel>();

    public int getPredictedClassLabel() {
    	return predictedClassLabel;
    }

    public void predict(int[] timeseries) {
        final int M = timeseries.length;
        int[] observationSequence = new int[M];

        for (int i = 0; i < M; i++) {
            observationSequence[i] = timeseries[i];
        }

        if (classLikelihoods.length != numClasses) {
            classLikelihoods = new double[numClasses];
        }
        if (classDistances.length != numClasses) {
            classDistances = new double[numClasses];
        }

        bestDistance = -99e+99;
        int bestIndex = 0;
        double sum = 0;
        for (int k = 0; k < numClasses; k++) {
            classDistances[k] = models.get(k).predict(observationSequence);

            // Set the class likelihood as the antilog of the class distances
            classLikelihoods[k] = antilog(classDistances[k]);

            // The loglikelihood values are negative so we want the values
            // closest to 0
            if (classDistances[k] > bestDistance) {
                bestDistance = classDistances[k];
                bestIndex = k;
            }

            sum += classLikelihoods[k];
        }

        // Turn the class distances into proper likelihoods
        for (int k = 0; k < numClasses; k++) {
            classLikelihoods[k] /= sum;
        }

        maxLikelihood = classLikelihoods[bestIndex];
        predictedClassLabel = classLabels[bestIndex];

        if (useNullRejection) {
            if (maxLikelihood > nullRejectionThresholds[bestIndex]) {
                predictedClassLabel = classLabels[bestIndex];
            } else {
                predictedClassLabel = 0;
            }
        }
    }

    private double antilog(double d) {
        return Math.exp(d);
    }

    void clear() {
        models.clear();
    }
    
    public void printTXT() {
        System.out.println("UseNullRejection: 0\n"
                + "NumClasses: 7\n"
                + "NullRejectionThresholds:  0 0 0 0 0 0 0\n"
                + "ClassLabels:  1 2 3 4 5 6 7");
        for (int k = 0; k < numClasses; k++) {
            System.out.println("Model_ID: " + (k+1));
            models.get(k).printTXT();
        }
        System.out.println("\n==========================");
    }
}
