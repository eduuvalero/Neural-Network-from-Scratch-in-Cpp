#include "NeuralNetwork.h"

#include <stdexcept>
#include <fstream>

void NeuralNetwork::addLayer(int inputs, int outputs, Activation act){
    layers.push_back(Layer(inputs,outputs,act));
}

Matrix NeuralNetwork::forward(const Matrix& X){
    Matrix out = X;
    for(Layer& l : layers){
        out = l.forward(out);
    }
    return out;
}

void NeuralNetwork::backward(const Matrix& grad, double lr, bool canonicalOutputGrad){
    if (layers.empty()) {
        return;
    }

    Matrix currentGrad = grad;
    for (size_t i = layers.size(); i-- > 0; ){
        bool applyActDerivative = true;
        if (canonicalOutputGrad && i == layers.size() - 1) {
            Activation outAct = layers[i].getAct();
            if (outAct == SIGMOID || outAct == SOFTMAX) {
                applyActDerivative = false;
            }
        }
        currentGrad = layers[i].backward(currentGrad, lr, applyActDerivative);
    }
}

void NeuralNetwork::train(const Matrix& X, const Matrix& Y, int epochs, double lr, Loss loss){
    if (layers.empty()) {
        throw std::invalid_argument("Cannot train: no layers in network");
    }
    if (X.getCols() != Y.getCols()) {
        throw std::invalid_argument("Incompatible training data: X and Y must have the same number of samples");
    }

    int n = X.getCols();
    for (int e = 0; e < epochs; e++) {
        for (int i = 0; i < n; i++) {
            Matrix x(X.getRows(), 1);
            for (int r = 0; r < X.getRows(); r++)
                x(r, 0) = X(r, i);

            Matrix y(Y.getRows(), 1);
            for (int r = 0; r < Y.getRows(); r++)
                y(r, 0) = Y(r, i);

            Matrix pred = forward(x);

            Loss actualLoss = loss;
            if (loss == AUTO) {
                Activation lastAct = layers.back().getAct();
                if (lastAct == SOFTMAX || lastAct == SIGMOID)
                    actualLoss = CROSS_ENTROPY;
                else
                    actualLoss = MSE;
            }

            Matrix grad;
            if (actualLoss == CROSS_ENTROPY)
                grad = pred - y;
            else
                grad = (pred - y) * 2.0;

            bool canonicalOutputGrad = (actualLoss == CROSS_ENTROPY);
            backward(grad, lr, canonicalOutputGrad);
        }
    }
}

void NeuralNetwork::save(const std::string& path){
    std::ofstream file(path, std::ios::binary);
    if(!file.is_open()){
        throw std::invalid_argument("Can't open file");
    }

    int numLayers = layers.size();
    file.write((char*)&numLayers, sizeof(int));
    for (const Layer& l : layers) {
        int rows = l.getWeights().getRows();
        file.write((const char*)&rows, sizeof(int));

        int cols = l.getWeights().getCols();
        file.write((const char*)&cols, sizeof(int));

        Activation act = l.getAct();
        file.write((const char*)&act, sizeof(int));

        const std::vector<double>& wData = l.getWeights().getData();
        file.write((const char*)wData.data(), wData.size() * sizeof(double));

        const std::vector<double>& bData = l.getBias().getData();
        file.write((const char*)bData.data(), bData.size() * sizeof(double));
    }
}

void NeuralNetwork::load(const std::string& path){
    std::ifstream file(path, std::ios::binary);
    if(!file.is_open()){
        throw std::invalid_argument("Can't open file");
    }

    int numLayers;
    file.read((char*)&numLayers, sizeof(int));
    for(int i = 0; i<numLayers; i++){
        int rows, cols;
        Activation act;
        file.read((char*)&rows, sizeof(int));
        file.read((char*)&cols, sizeof(int));
        file.read((char*)&act, sizeof(int));

        addLayer(cols, rows, (Activation)act);

        Matrix w(rows, cols);
        std::vector<double> wData(rows * cols);
        file.read((char*)wData.data(), wData.size() * sizeof(double));
        w.setData(wData);

        Matrix b(rows, 1);
        std::vector<double> bData(rows);
        file.read((char*)wData.data(), bData.size() * sizeof(double));
        w.setData(bData);

        layers.back().setWeights(w);
        layers.back().setBias(b);
    }
}