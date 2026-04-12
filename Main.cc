#include <iostream>
#include "NeuralNetwork.h"

int main() {
    Matrix X(2, 4);
    X(0,0)=0; X(1,0)=0;
    X(0,1)=0; X(1,1)=1;
    X(0,2)=1; X(1,2)=0;
    X(0,3)=1; X(1,3)=1;

    Matrix Y(1, 4);
    Y(0,0)=0;
    Y(0,1)=0;
    Y(0,2)=0;
    Y(0,3)=1;

    NeuralNetwork model;
    model.addLayer(2, 4, RELU);
    model.addLayer(4, 3, RELU);
    model.addLayer(3, 1);

    model.train(X, Y, 10000, 0.03);

    std::cout << "Predicciones:\n";
    for (int i = 0; i < 4; i++) {
        Matrix x(2, 1);
        x(0,0) = X(0,i);
        x(1,0) = X(1,i);
        model.forward(x).print();
    }

    std::cout << "Esperado:\n";
    Y.print();

    return 0;
}