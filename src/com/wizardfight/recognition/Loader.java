/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package com.wizardfight.recognition;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

/**
 *
 * @author Konstantin
 */
public class Loader {

    final int NUM_SYMBOLS = 20; // 10 - default
    public KMeansQuantizer quantizer = new KMeansQuantizer(NUM_SYMBOLS);
    public HMM hmm = new HMM();

    public boolean loadHMMFromFile(String file) throws IOException {
        hmm.clear();

        BufferedReader reader;

        try {
            reader = new BufferedReader(new FileReader(file));
        } catch (FileNotFoundException ex) {
            System.err
                    .println("loadDatasetFromFile(String filename) - FILE NOT OPEN!");
            return false;
        }

        String word;
        double value;

        // Load if the number of clusters
        word = reader.readLine();
        if (!word.contains("UseNullRejection:")) {
            System.err
                    .println("loadBaseSettingsFromFile(fstream &file) - Failed to read UseNullRejection header!");
            reader.close();
            hmm.clear();
            return false;
        }
        String[] buf = word.split("\\s+");
        hmm.useNullRejection = (Integer.parseInt(buf[1]) == 1);

        // If the model is trained then load the model settings
        // Load the number of classes
        word = reader.readLine();
        if (!word.contains("NumClasses:")) {
            System.err
                    .println("loadBaseSettingsFromFile(fstream &file) - Failed to read NumClasses header!");
            hmm.clear();
            reader.close();
            return false;
        }
        buf = word.split("\\s+");
        hmm.numClasses = Integer.parseInt(buf[1]);

        // Load the null rejection thresholds
        word = reader.readLine();
        if (!word.contains("NullRejectionThresholds:")) {
            System.err
                    .println("loadBaseSettingsFromFile(fstream &file) - Failed to read NullRejectionThresholds header!");
            hmm.clear();
            reader.close();
            return false;
        }
        hmm.nullRejectionThresholds = new double[hmm.numClasses];
        buf = word.split("\\s+");
//            System.out.println(word);
        for (int i = 0; i < hmm.nullRejectionThresholds.length; i++) {

            hmm.nullRejectionThresholds[i] = Integer.parseInt(buf[i + 1]);
        }
        // Load the class labels
        word = reader.readLine();
        if (!word.contains("ClassLabels:")) {
            System.err
                    .println("loadBaseSettingsFromFile(fstream &file) - Failed to read ClassLabels header!");
            hmm.clear();
            reader.close();
            return false;
        }
        hmm.classLabels = new int[hmm.numClasses];
        buf = word.split("\\s+");
        for (int i = 0; i < hmm.classLabels.length; i++) {
            hmm.classLabels[i] = Integer.parseInt(buf[i + 1]);
        }

        // If the HMM has been trained then load the hmm.models
        // Resize the buffer
        hmm.models.ensureCapacity(hmm.numClasses);

        // Load each of the K classes
        for (int k = 0; k < hmm.numClasses; k++) {
            int modelID;
            word = reader.readLine();
            if (!word.contains("Model_ID:")) {
                System.err
                        .println("loadModelFromFile( fstream &file ) - Could not find model ID for the "
                                + (k + 1) + "th model");
                reader.close();
                return false;
            }
            buf = word.split("\\s+");
            modelID = Integer.parseInt(buf[1]);

            if (modelID - 1 != k) {
                System.err
                        .println("loadModelFromFile( fstream &file ) - Model ID does not match the current class ID for the "
                                + (k + 1) + "th model");
                reader.close();
                return false;
            }
            word = reader.readLine();
            if (!word.contains("NumStates:")) {
                System.err
                        .println("loadModelFromFile( fstream &file ) - Could not find the NumStates for the "
                                + (k + 1) + "th model");
                reader.close();
                return false;
            }
            buf = word.split("\\s+");
            hmm.models.add(k, new HiddenMarkovModel());
            hmm.models.get(k).numStates = Integer.parseInt(buf[1]);

            hmm.models.get(k).a = new double
                    [ hmm.models.get(k).numStates ]
                    [ hmm.models.get(k).numStates ];
            hmm.models.get(k).pi = new double[hmm.models.get(k).numStates];

            word = reader.readLine();
            // Load the A, B and Pi matrices
            if (!word.contains("A:")) {
                System.err
                        .println("loadModelFromFile( fstream &file ) - Could not find the A matrix for the "
                                + (k + 1) + "th model.");
                reader.close();
                return false;
            }

            for (int i = 0; i < hmm.models.get(k).numStates; i++) {
                word = reader.readLine();
//                    System.out.println(word);
                buf = word.split("\\s+");
                for (int j = 0; j < hmm.models.get(k).numStates; j++) {
                    value = Double.parseDouble(buf[j]);
                    hmm.models.get(k).a[i][j] = value;
                }
            }
            word = reader.readLine();
            if (!word.contains("B:")) {
                System.err
                        .println("loadModelFromFile( fstream &file ) - Could not find the B matrix for the "
                                + (k + 1) + "th model.");
                reader.close();
                return false;
            }

            // Load B
            // word = reader.readLine();
            hmm.models.get(k).b = new double 
                    [hmm.models.get(k).numStates][NUM_SYMBOLS];
            for (int i = 0; i < hmm.models.get(k).numStates; i++) {
                word = reader.readLine();
                buf = word.split("\\s+");
                for (int j = 0; j < NUM_SYMBOLS; j++) {
                    value = Double.parseDouble(buf[j]);
                    hmm.models.get(k).b[i][j] = value;
                }
            }

            word = reader.readLine();
            if (!word.contains("Pi:")) {
                System.err
                        .println("loadModelFromFile( fstream &file ) - Could not find the Pi matrix for the "
                                + (k + 1) + "th model.");
                reader.close();
                return false;
            }

            // Load Pi
            word = reader.readLine();
            buf = word.split("\\s+");
            for (int i = 0; i < hmm.models.get(k).numStates; i++) {
                value = Double.parseDouble(buf[i]);
                hmm.models.get(k).pi[i] = value;
            }
        }

        hmm.maxLikelihood = 0;
        hmm.bestDistance = 0;
        hmm.classLikelihoods = new double[hmm.numClasses];
        hmm.classDistances = new double[hmm.numClasses];
        reader.close();
        return true;
    }

    public boolean loadQuantizerFromFile(String file) throws IOException {

        quantizer.numClusters = 0;
        quantizer.clusters = null;
        quantizer.quantizationDistances.clear();

        BufferedReader reader;

        try {
            reader = new BufferedReader(new FileReader(file));
        } catch (FileNotFoundException ex) {
            System.err
                    .println("loadDatasetFromFile(String filename) - FILE NOT OPEN!");
            return false;
        }

        String word;
        quantizer.numInputDimensions = 3;

        word = reader.readLine();
        if (!word.contains("NumClusters:")) {
            System.err
                    .println("loadModelFromFile(fstream &file) - Failed to load NumClusters!");
            reader.close();
            return false;
        }
        quantizer.numClusters = Integer.parseInt(word.split("\\s+")[1]);

        quantizer.clusters = new double[quantizer.numClusters][quantizer.numInputDimensions];
        word = reader.readLine();
        if (!word.contains("Clusters:")) {
            System.err
                    .println("loadModelFromFile(fstream &file) - Failed to load Clusters!");
            reader.close();
            return false;
        }

        String[] buf;
        for (int k = 0; k < quantizer.numClusters; k++) {
            word = reader.readLine();
            buf = word.split("\\s+");
            for (int j = 0; j < quantizer.numInputDimensions; j++) {
                quantizer.clusters[k][j] = Double.parseDouble(buf[j]);
            }
        }
        
        quantizer.quantizationDistances.ensureCapacity(quantizer.numClusters);
        reader.close();
        return true;
    }
}
