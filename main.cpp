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


double db_dot_product(std::vector<double> x, std::vector<double> y) {
    
    float sum = 0;
    for(int i = 0; i < x.size(); i++) {
        sum += x[i]*y[i];
    }
    
    return sum;
}

class LayerDense {
    private:
        std::vector<std::vector<double>> weights;
        std::vector<std::vector<double>> biases;
        std::vector<std::vector<double>> outputs;
    public:
        LayerDense(double n_inputs, double n_neurons) {
            // init weights
            // weights should have shape (n_neurons, n_inputs)
            // normally ( if we were doing actual matrix multiplication ) to do a dot product
            // the shapes must be opposite (2, 1) (1, 2) to result in a list of single values
            // but for my implementation they must be the same :D
            for(int n = 0; n < n_neurons; n++) {
                std::vector<double> weightRow;
                for (int i = 0; i < n_inputs; i++) {
                    // give each weight a value from -1 to 1 with 3 decimals of presicion
                    // we make a number like 10123 then divide by 10000 to get .10123
                    // then subtract by half the max ( plus .0 to make a double ) 
                    // so we get an equal amount of negative and positive weights
                    weightRow.push_back(((std::rand() % 20001) - 10000.0)/10000);
                }
                weights.push_back(weightRow);
            }

            for (std::vector<double> v : weights) {
                for (double d : v) {
                    std::cout << std::to_string(d) + ", ";
                }
                std::cout << "\n";
            }

            // init biases
            // shape of bias matrix is (1, n_neurons) set them all to 0
            // creat new scope so temp row var doesn't leak into global namespace
            {
                std::vector<double> biasesRow;
                for (int i = 0; i < n_neurons; i++) {
                    biasesRow.push_back(0.0);
                }
                biases.push_back(biasesRow);
            }

        }

        std::vector<std::vector<double>> forward(std::vector<std::vector<double>> inputs) {
            // calculate the outputs for each neuron in layer
            int neuronCount = 0;

            for(std::vector<double> w : weights) {
                std::vector<double> outputForNeuron;
                // AND THEN for each neuron in the layer we calculate each output in the batch
                for (std::vector<double> i : inputs) {
                    outputForNeuron.push_back(db_dot_product(w, i) + biases[0][neuronCount]);
                }

                // now that we have the outputs for the neuron append them to the output list
                outputs.push_back(outputForNeuron);

                neuronCount++;
            }

            return outputs;
        }
};

int main(int argc, const char * argv[]) {
    std::cout << "Starting.... \n";
    
    std::vector<std::vector<double>> X = {{1, 2, 3, 2.5}, {2.0, 5.0, -1.0, 2.0}, {-1.5, 2.7, 3.3, -0.8}};

    LayerDense ld(4, 5);

    std::vector<std::vector<double>> output = ld.forward(X);
    std::cout << "---- OUTPUT ----" << std::endl;

    for (std::vector<double> v : output) {
        for (double d : v) {
            std::cout << std::to_string(d) + ", ";
        }
        std::cout << "\n";
    }

    return 0;
}

