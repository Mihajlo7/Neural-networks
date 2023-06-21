/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package breast_cancer_neuroph;

import org.neuroph.core.NeuralNetwork;

/**
 *
 * @author Mihajlo
 */
public class Trainig {
    public double msne;
    public NeuralNetwork neuralNet;

    public Trainig(double msne, NeuralNetwork neuralNet) {
        this.msne = msne;
        this.neuralNet = neuralNet;
    }

    public Trainig() {
    }
    
    
}
