/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Main.java to edit this template
 */
package breast_cancer_neuroph;

import java.util.ArrayList;
import java.util.List;
import neuroph_template.NeurophTemplate;
import org.neuroph.core.data.DataSet;
import org.neuroph.eval.ClassifierEvaluator;
import org.neuroph.eval.Evaluation;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.data.norm.MaxNormalizer;
import org.neuroph.util.data.norm.Normalizer;

/**
 *
 * @author Mihajlo
 */
public class BreastCancerNeuroph implements NeurophTemplate{

    /**
     * @param args the command line arguments
     */
    List<Trainig> trainings=new ArrayList<>();
    int input=30;
    int output=1;
    double[] learningRates={0.2,0.4,0.6};
    double learningRate;
    int[] hiddenNeurons={10,20,30};
    int hiddenNeuron;
    DataSet train;
    DataSet test;
    int numberOfIterations;
    int numberOfTrainings;
    public static void main(String[] args) {
       new BreastCancerNeuroph().run();
    }

    private void run() {
        System.out.println("Creating data set ...");
        DataSet dataSet=loadDataSet();
        System.out.println("Normalizing and Shuffling ...");
        dataSet=preprocessDataSet(dataSet);
        System.out.println("Spliting data ...");
        DataSet[] trainAndTest=trainTestSplit(dataSet);
        train=trainAndTest[0];
        test=trainAndTest[1];
        
        for(double lr:learningRates){
            for(int hn:hiddenNeurons){
                System.out.println("Creating neuralNet with lr: "+lr+"   and hn: "+hn);
                hiddenNeuron=hn;
                learningRate=lr;
                MultiLayerPerceptron neuralNet=createNeuralNetwork();
                trainNeuralNetwork(neuralNet, train);
            }
        }
        System.out.println("Iterations: "+numberOfIterations+"  Trainings: "+numberOfTrainings);
        System.out.println("Avarage iterations: "+(double)numberOfIterations/numberOfTrainings);
        System.out.println("Saving neuralNet ...");
        saveBestNetwork();
    }

    @Override
    public DataSet loadDataSet() {
        DataSet dataSet=DataSet.createFromFile("data/breast_cancer_data.csv", input, output,",");
        return dataSet;
    }

    @Override
    public DataSet preprocessDataSet(DataSet ds) {
        Normalizer normalizer=new MaxNormalizer(ds);
        normalizer.normalize(ds);
        ds.shuffle();
        return ds;
    }

    @Override
    public DataSet[] trainTestSplit(DataSet ds) {
        return ds.split(0.65,0.35);
    }

    @Override
    public MultiLayerPerceptron createNeuralNetwork() {
        return new MultiLayerPerceptron(input,hiddenNeuron,output);
    }

    @Override
    public MultiLayerPerceptron trainNeuralNetwork(MultiLayerPerceptron mlp, DataSet ds) {
        mlp.setLearningRule(new MomentumBackpropagation());
        MomentumBackpropagation learningRule=(MomentumBackpropagation)mlp.getLearningRule();
        learningRule.setMomentum(0.7);
        learningRule.setLearningRate(learningRate);
        learningRule.addListener((event)->{
            MomentumBackpropagation mbp=(MomentumBackpropagation)event.getSource();
            System.out.println(mbp.getCurrentIteration()+". iteration | Total Error: "+mbp.getTotalNetworkError());
        });
        learningRule.setMaxIterations(2000);
        
        mlp.learn(ds);
        
        evaluate(mlp, test);
        
        numberOfTrainings++;
        numberOfIterations+=learningRule.getCurrentIteration();
        return mlp;
    }

    @Override
    public void evaluate(MultiLayerPerceptron mlp, DataSet ds) {
        Evaluation eval=new Evaluation();
        eval.addEvaluator(new ClassifierEvaluator.Binary(0.5));
        eval.evaluate(mlp, test);
        
        System.out.println("MSNE: "+eval.getMeanSquareError());
        
        trainings.add(new Trainig(eval.getMeanSquareError(),mlp));
    }

    @Override
    public void saveBestNetwork() {
        Trainig min=trainings.get(0);
        for(int i=1;i<trainings.size();i++){
            if(trainings.get(i).msne<min.msne){
                min=trainings.get(i);
            }
        }
        min.neuralNet.save("neuralNet.nn");
    }
    
}
