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
#include <numeric>
#include <time.h>
#include "data.hpp"


double db_dot_product(std::vector<double> x, std::vector<double> y) {
    
    /*
    I found out about inner_product, rip for my custom solution :(
    for(int i = 0; i < x.size(); i++) {
        sum += x[i]*y[i];
        std::cout << sum << " = " << x[i] << " * " << y[i] << "\n";
    }
    std::cout << std::endl; 
    */
    
    return std::inner_product(std::begin(x), std::end(x), std::begin(y), 0.0);
}

class ReLU {
    public:
        std::vector<std::vector<double>> forward(std::vector<std::vector<double>> input) {
            std::vector<std::vector<double>> output;

            for (std::vector<double> v : input) {
                std::vector<double> outputRow;
                for(double d : v) {
                    if (d > 0) {
                        outputRow.push_back(d);
                    } else {
                        outputRow.push_back(0);
                    }
                }

                output.push_back(outputRow);
            }

            return output;
        }
};

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
                    // give each weight a value from -.1 to .1 with 3 decimals of presicion
                    // we make a number like 10123 then divide by 100000 to get .010123
                    // then subtract by half the max ( plus .0 to make a double ) 
                    // so we get an equal amount of negative and positive weights
                    weightRow.push_back(((std::rand() % 20001) - 10000.0)/100000);
                }
                weights.push_back(weightRow);
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

            // construct our output with shape x axis size of batch size and y axis of n_neurons
            for(std::vector<double> i : inputs) {
                std::vector<double> outputRow;

                int neuronCount = 0;
                for (std::vector<double> w : weights) {
                    outputRow.push_back(db_dot_product(w, i) + biases[0][neuronCount]);

                    neuronCount++;
                }

                outputs.push_back(outputRow);
            }

            return outputs;
        }
};

int main(int argc, const char * argv[]) {
    std::cout << "Starting.... \n";
    
    std::srand(std::time(NULL));
    
    Data d;

    std::vector<std::vector<double>> X = d.getDataX();
    std::vector<double> y = d.getDataY();

    LayerDense ld(2, 5);
    ReLU activation;
    
    std::vector<std::vector<double>> output = activation.forward(ld.forward(X));

    std::cout << "---- OUTPUT ----" << std::endl;

    for (std::vector<double> v : output) {    
        for (double d : v) {
            std::cout << d << ", ";
        }
        std::cout << std::endl;
    }

    return 0;
}

