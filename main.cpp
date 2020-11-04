//
//  main.cpp
//  NeuralNet
//
//  Created by Jón Salenger on 11/3/20.
//  Copyright © 2020 Jón Salenger. All rights reserved.
//

#include <iostream>
#include <vector>
#include <string>

double db_dot_product(std::vector<double> x, std::vector<double> y);

int main(int argc, const char * argv[]) {
    std::cout << "Starting.... \n";
    
    std::vector<double> inputs = {1, 2, 3, 2.5};
    std::vector<std::vector<double>> weights = {{0.2, 0.8, -0.5, 1.0}, {0.5, -0.91, 0.26, -0.5}, {-0.26, -0.27, 0.17, 0.87}};
    std::vector<double> biases = {2, 3, 0.5};
        
    // calculate the output of each neuron
    // neurons are calculated using weights[0]*inputs[0]+weights[1]*inputs[1]...+bias
    // 1 bias per neuron SO biases.size() == # of neurons
    std::vector<double> neuron_outputs;
    // n is the neuron
    for(int n = 0; n < biases.size(); n++) {
        // dot product is the math term for input*weight
        // compute that over all inputs and weights and then just add bias
        neuron_outputs.push_back(db_dot_product(inputs, weights[n]) + biases[n]);
    }
    
    for(double i : neuron_outputs) {
        std::cout << std::to_string(i) << std::endl;
    }

    return 0;
}

double db_dot_product(std::vector<double> x, std::vector<double> y) {
    
    float sum = 0;
    for(int i = 0; i < x.size(); i++) {
        sum += x[i]*y[i];
    }
    
    return sum;
}
