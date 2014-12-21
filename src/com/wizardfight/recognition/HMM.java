package com.wizardfight.recognition;
import java.io.Serializable;
import java.util.ArrayList;

/**
 * This class acts as the main interface for using a Hidden Markov Model.
 */
public class HMM implements Serializable {

    private static final long serialVersionUID = 2L;
    // Variables for all the HMMs
    protected boolean trained = false;
    protected boolean useScaling = false;
    protected boolean useNullRejection = false;

    protected int delta = 1; // The number of states a model can move to in a
    // HMMModelTipes.LEFTRIGHT model
    protected int maxNumIter = 100; // The maximum number of iter allowed during
    // the full training
    protected int numClasses;
    protected int numInputDimensions = 0;
    protected int numRandomTrainingIterations;
    protected int numStates = 5; // The number of states for each model
    protected int numSymbols = 10; // The number of symbols for each model
    protected int predictedClassLabel;

    protected double bestDistance;
    protected double maxLikelihood;
    protected double minImprovement = 1.0e-2; // The minimum improvement value
    // for each model during training
    protected int[] classLabels = new int[0];
    protected double[] classLikelihoods = new double[0];
    protected double[] classDistances = new double[0];
    protected double[] nullRejectionThresholds;

    protected HMMModelTypes modelType = HMMModelTypes.LEFTRIGHT;

    protected ArrayList<HiddenMarkovModel> models = new ArrayList<HiddenMarkovModel>();

    public int getPredictedClassLabel() {
        if (trained) {
            return predictedClassLabel;
        }
        return 0;
    }

    /**
     * This predicts the class of the timeseries.
     *
     * @param MatrixDouble timeSeries: the input timeseries to classify
     * @return returns true if the prediction was performed, false otherwise
     */
    public boolean predict(MatrixDouble timeseries) {
        // Covert the matrix double to observations
        final int M = timeseries.rows;
        int[] observationSequence = new int[M];

        for (int i = 0; i < M; i++) {
            observationSequence[i] = (int) timeseries.dataPtr[i][0];
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

        return true;
    }

    private double antilog(double d) {
        return Math.exp(d);
    }

    protected void clear() {
        models.clear();
    }
    
    public void print() {
        System.out.println(trained + " " + useScaling + " " + useNullRejection);
        System.out.println(delta + " " + maxNumIter + " " + numClasses);
        System.out.println(numInputDimensions + " " + numRandomTrainingIterations + " " + numStates);
        System.out.println(numSymbols + " " + predictedClassLabel);
        System.out.println(bestDistance + " " + maxLikelihood + " " + minImprovement);
        for(int i=0; i<classLabels.length; i++) {
            System.out.print(classLabels[i] + " ");
        }
        System.out.println();
        
        for(int i=0; i<classLikelihoods.length; i++) {
            System.out.print(classLikelihoods[i] + " ");
        }
        System.out.println();
        
        for(int i=0; i<classDistances.length; i++) {
            System.out.print(classDistances[i] + " ");
        }
        System.out.println();
        
        for(int i=0; i<nullRejectionThresholds.length; i++) {
            System.out.print(nullRejectionThresholds[i] + " ");
        }
        System.out.println();
        
        System.out.println(modelType);
        
        for(HiddenMarkovModel hmm : models) {
            hmm.print();
        }
        System.out.println("\n==========================");
    }
    
    public void printTXT() {
        System.out.println("HMM_MODEL_FILE_V2.0\n"
                + "Trained: 1\n"
                + "UseScaling: 0\n"
                + "NumInputDimensions: 1\n"
                + "NumOutputDimensions: 0\n"
                + "NumTrainingIterationsToConverge: 0\n"
                + "MinNumEpochs: 0\n"
                + "MaxNumEpochs: 100\n"
                + "ValidationSetSize: 20\n"
                + "LearningRate: 0.1\n"
                + "MinChange: 1e-005\n"
                + "UseValidationSet: 0\n"
                + "RandomiseTrainingOrder: 1\n"
                + "UseNullRejection: 0\n"
                + "ClassifierMode: 1\n"
                + "NullRejectionCoeff: 5\n"
                + "NumClasses: 7\n"
                + "NullRejectionThresholds:  0 0 0 0 0 0 0\n"
                + "ClassLabels:  1 2 3 4 5 6 7\n"
                + "NumStates: 4\n"
                + "NumSymbols: 20\n"
                + "ModelType: 1\n"
                + "Delta: 1\n"
                + "NumRandomTrainingIterations: 20");
        for (int k = 0; k < numClasses; k++) {
            System.out.println("Model_ID: " + (k+1));
            models.get(k).printTXT();
        }
        System.out.println("\n==========================");
    }
}
