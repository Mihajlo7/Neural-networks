
import java.util.ArrayList;
import java.util.List;
import neuroph_template.NeurophTemplate;
import org.neuroph.core.data.DataSet;
import org.neuroph.eval.ClassifierEvaluator;
import org.neuroph.eval.Evaluation;
import org.neuroph.eval.classification.ClassificationMetrics;
import org.neuroph.eval.classification.ConfusionMatrix;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.data.norm.MaxNormalizer;
import org.neuroph.util.data.norm.Normalizer;

/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */

/**
 *
 * @author Mihajlo
 */
public class NeurophDiabetes implements NeurophTemplate{

    List<Training> trainings=new ArrayList<>();
    int input=8;
    int output=1;
    int[] hiddenNeurons={20,10};
    double[] learningRates={0.2,0.4,0.6};
    double learningRate;
    int numberOfIterations=0;
    int numberOfTrainings=0;
    DataSet train;
    DataSet test;
    public static void main(String[] args) {
        new NeurophDiabetes().run();
    }
    
    private void run() {
        DataSet dataSet=loadDataSet();
        dataSet=preprocessDataSet(dataSet);
        
        DataSet[] trainAndTest=trainTestSplit(dataSet);
        train=trainAndTest[0];
        test=trainAndTest[1];
        
        
        for(double lr:learningRates){
            learningRate=lr;
            System.out.println("Creating data set with lr: "+lr);
            MultiLayerPerceptron neuralNet=createNeuralNetwork();
            trainNeuralNetwork(neuralNet, train);
        }
        System.out.println("iterations:"+numberOfIterations+"  trainigs: "+numberOfTrainings);
        System.out.println("Avarage iterations: "+(double)numberOfIterations/numberOfTrainings);
        System.out.println("Saving ...");
        saveBestNetwork();
        System.out.println("All trainigs completed.");
    }

    @Override
    public DataSet loadDataSet() {
        System.out.println("Creating data sets ...");
        DataSet dataSet=DataSet.createFromFile("data/diabetes_data.csv", input, output, ",");
        return dataSet;
    }

    @Override
    public DataSet preprocessDataSet(DataSet ds) {
        System.out.println("Normalizing ...");
        Normalizer normalizer=new MaxNormalizer(ds);
        normalizer.normalize(ds);
        System.out.println("Shuffling ...");
        ds.shuffle();
        return ds;
    }

    @Override
    public DataSet[] trainTestSplit(DataSet ds) {
        System.out.println("Spliting data ...");
        return ds.split(0.7,0.3);
    }

    @Override
    public MultiLayerPerceptron createNeuralNetwork() {
        return new MultiLayerPerceptron(input,20,10,output);
    }

    @Override
    public MultiLayerPerceptron trainNeuralNetwork(MultiLayerPerceptron mlp, DataSet ds) {
        mlp.setLearningRule(new BackPropagation());
        BackPropagation bp=(BackPropagation)mlp.getLearningRule();
        bp.setLearningRate(learningRate);
        bp.setMaxError(0.07);
        bp.addListener((event)->{
            BackPropagation bpp=(BackPropagation)event.getSource();
            System.out.println(bpp.getCurrentIteration()+". iteration | Total Error: "+bpp.getTotalNetworkError());
        });
        mlp.learn(train);
        
        evaluate(mlp, test);
        numberOfTrainings++;
        numberOfIterations+=bp.getCurrentIteration();
        return mlp;
    }

    @Override
    public void evaluate(MultiLayerPerceptron mlp, DataSet ds) {
        Evaluation eval=new Evaluation();
        eval.addEvaluator(new ClassifierEvaluator.Binary(0.5));
        eval.evaluate(mlp, ds);
        
        ConfusionMatrix cm=eval.getEvaluator(ClassifierEvaluator.Binary.class).getResult();
        ClassificationMetrics[] metrics=ClassificationMetrics.createFromMatrix(cm);
        ClassificationMetrics.Stats avarage=ClassificationMetrics.average(metrics);
        
        System.out.println("Accuracy: "+avarage.accuracy);
        
        trainings.add(new Training(avarage.accuracy,mlp));
    }

    @Override
    public void saveBestNetwork() {
        Training max=trainings.get(0);
        for(int i=1;i<trainings.size();i++){
            if(max.accuracy<trainings.get(i).accuracy){
                max=trainings.get(i);
            }
        }
        max.neuralNet.save("neuralNet.nn");
    }

   
    

    
    
}
