package com.wizardfight.recognition;

import java.io.*;
import java.util.*;
import com.wizardfight.Shape;
import com.wizardfight.components.Vector3d;

public class Recognizer {

    static int sum;
    private static KMeansQuantizer quantizer;
    private static HMM hmm;

    public static void init() {
        // Load quantizer from serialized file
        try {
            ObjectInputStream is = new ObjectInputStream(
                    new FileInputStream(new File("HMMQuantizer.ser")));
            quantizer = (KMeansQuantizer) is.readObject();
            is.close();
        } catch (Exception ex) {
            System.err.println("ERROR: Failed to load quantizer! " + ex);
        }

        // Create a new HMM instance
        hmm = new HMM();
        // Load the HMM model from a file
        try {
            ObjectInputStream is = new ObjectInputStream(
                    new FileInputStream(new File("HMMModel.ser")));
            hmm = (HMM) is.readObject();
            is.close();
        } catch (Exception ex) {
            System.err.println("ERROR: Failed to load hmm! " + ex);
        }

    }

    public static Shape recognize(ArrayList<Vector3d> records) {
    	long startStamp = System.currentTimeMillis();
        quantizer.refresh();

        int[] timeSeries = new int[ records.size() ];
        double[] rec = new double[3];
        
        for (int j = 0; j < timeSeries.length ; j++) {
        	rec[0] = records.get(j).x;
        	rec[1] = records.get(j).y;
        	rec[2] = records.get(j).z;
            timeSeries[j] = quantizer.quantize(rec);
        }
        
        hmm.predict(timeSeries);

        System.out.println("Time: " + (System.currentTimeMillis()-startStamp) + " ms");
        return getShape(hmm.getPredictedClassLabel());
    }

    static Shape getShape(int val) {
        Shape s;
        switch (val) {
            case 1:
                s = Shape.CIRCLE;
                break;
            case 2:
                s = Shape.CLOCK;
                break;
            case 3:
                s = Shape.PI;
                break;
            case 4:
                s = Shape.SHIELD;
                break;
            case 5:
                s = Shape.TRIANGLE;
                break;
            case 6:
                s = Shape.V;
                break;
            case 7:
                s = Shape.Z;
                break;
            default:
                s = Shape.FAIL;
        }
        return s;
    }

    // Converting class id into label
    static void printLabel(int val) {
        System.out.print("Figure is ");
        switch (val) {
            case 1:
                System.out.print("circle");
                break;
            case 2:
                System.out.print("clock");
                break;
            case 3:
                System.out.print("pi");
                break;
            case 4:
                System.out.print("shield");
                break;
            case 5:
                System.out.print("triangle");
                break;
            case 6:
                System.out.print("v");
                break;
            case 7:
                System.out.print("z");
                break;
        }
        System.out.println();
    }
    
    public static KMeansQuantizer getQuantizerFromFile(String path) {
        Loader loader = new Loader();
        // Load quantizer from serialized file
        try {
            loader.loadQuantizerFromFile(path);
            return loader.quantizer;
        } catch (Exception ex) {
            System.err.println("ERROR: Failed to write quantizer! " + ex);
            return null;
        }
    }
    
    public static HMM getHMMFromFile(String path) {
        Loader loader = new Loader();
        try {
            loader.loadHMMFromFile(path);
            return loader.hmm;
        } catch (Exception ex) {
            System.err.println("ERROR: Failed to write hmm! ");
            ex.printStackTrace();
            return null;
        }
    }
    
    public static void writeSerialized(String quantizerPath, String hmmPath) {
        Loader loader = new Loader();
        // Load quantizer from serialized file
        try {
            ObjectOutputStream os = new ObjectOutputStream(
                    new FileOutputStream(new File("HMMQuantizer.ser")));
            loader.loadQuantizerFromFile(quantizerPath);
            os.writeObject(loader.quantizer);
            quantizer = loader.quantizer;
            os.close();
        } catch (Exception ex) {
            System.err.println("ERROR: Failed to write quantizer! " + ex);
        }

        // Load the HMM model from a file
        try {
            ObjectOutputStream os = new ObjectOutputStream(
                    new FileOutputStream(new File("HMMModel.ser")));
            loader.loadHMMFromFile(hmmPath);
            os.writeObject(loader.hmm);
            hmm = loader.hmm;
            os.close();
        } catch (Exception ex) {
            System.err.println("ERROR: Failed to write hmm! " + ex);
        }
    }

    public static Object readObject(File file) {
         try {
            ObjectInputStream is = new ObjectInputStream(
                    new FileInputStream(file));
            Object q = is.readObject();
            is.close();
            return q;
        } catch (Exception ex) {
            System.err.println("ERROR: Failed to write quantizer! " + ex);
            return null;
        }
    }
    public static ArrayList<Vector3d> getRecordsFromFile(File file) {
        ArrayList<Vector3d> recs = new ArrayList<Vector3d>();
        try {
            Scanner sc = new Scanner(file);

            while (sc.hasNext()) {
                String[] nums = sc.nextLine().split(" ");
                Vector3d rec = new Vector3d();
                rec.x = Double.parseDouble(nums[0]);
                rec.y = Double.parseDouble(nums[1]);
                rec.z = Double.parseDouble(nums[2]);
                recs.add(rec);
            }
            sc.close();
        } catch (FileNotFoundException e) {
            System.out.println("Records file not found");
        }
        return recs;
    }

    public static void main(String[] args) {
//        KMeansQuantizer q = (KMeansQuantizer) 
//                Recognizer.readObject(new File("hmm_quantizer.ser"));
//        KMeansQuantizer q3 = (KMeansQuantizer) 
//                Recognizer.readObject(new File("HMMQuantizer.ser"));
//        HMM hmm = (HMM) 
//                Recognizer.readObject(new File("hmm_model.ser"));
        HMM hmm3 = (HMM) 
                Recognizer.readObject(new File("HMMModel.ser"));
        hmm3.printTXT();
//        q3.print();
//        HMM hmm = Recognizer.getHMMFromFile("wf_hmm_model_new.txt");
//        hmm.printTXT();
//        
//        KMeansQuantizer q1 = Recognizer.getQuantizerFromFile("wf_hmm_quantizer_new.txt");
//        q1.print();
        Recognizer.writeSerialized("wf_hmm_quantizer_new.txt", "wf_hmm_model_new.txt");
    }
}
