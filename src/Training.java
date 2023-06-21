
import org.neuroph.core.NeuralNetwork;

/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */

/**
 *
 * @author Mihajlo
 */
public class Training {
    public double accuracy;
    public NeuralNetwork neuralNet;

    public Training() {
    }

    public Training(double accuracy, NeuralNetwork neuralNet) {
        this.accuracy = accuracy;
        this.neuralNet = neuralNet;
    }
    
    
}
