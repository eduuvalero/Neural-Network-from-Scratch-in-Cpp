#include "Layer.h"
#include "Optimizer.h"

#include <cmath>
#include <random>
#include <string>
#include <stdexcept>

Layer::Layer(int inputs, int outputs, Activation act, Inicialization init, double dropoutRate): W(inputs, outputs), b(1, outputs), act(act), init(init), dropoutRate(dropoutRate) {
    if (inputs <= 0 || outputs <= 0) {
        throw std::invalid_argument(
            "Layer::Layer invalid dimensions: " +
            std::to_string(inputs) + "x" + std::to_string(outputs)
        );
    }
    if (dropoutRate < 0.0 || dropoutRate >= 1.0) {
        throw std::invalid_argument("Layer::Layer dropout must be in [0, 1)");
    }

    initWeights();

    for (int i = 0; i < outputs; i++){
        b(0, i) = 0.0;
    }
}

void Layer::initWeights(){
    switch (init){
        case HE:
            initHe();
            break;
        case XAVIER:
            initXavier();
            break;
        case AUTO:
            if(act == RELU || act == LEAKY_RELU || act == NONE){
                initHe();
            }
            else{
                initXavier();
            }
            break;
    }
}

void Layer::initHe(){
    int m = W.getRows();
    int n = W.getCols();
    double stddev = sqrt(2.0 / m);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            W(i, j) = rand.normal(0.0, stddev);
        }
    }
}

void Layer::initXavier(){
    int m = W.getRows();
    int n = W.getCols();
    double limit = sqrt(6.0 / (m + n));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            W(i, j) = rand.uniform(-limit, limit);
        }
    }
}

double Layer::stableSigmoid(double x) {
    if (x >= 0.0) {
        double z = std::exp(-x);
        return 1.0 / (1.0 + z);
    }

    double z = std::exp(x);
    return z / (1.0 + z);
}

Matrix Layer::applyActivation(const Matrix& z) const {
    switch (act) {
        case RELU:
            return z.map([](double x) { return x > 0.0 ? x : 0.0; });
        case LEAKY_RELU:
            return z.map([](double x) { return x > 0.0 ? x : 0.01 * x; });
        case SIGMOID:
            return z.map([](double x) { return stableSigmoid(x); });
        case TANH:
            return z.map([](double x) { return std::tanh(x); });
        case SOFTMAX:
            return z.softmax();
        case NONE:
            return z;
        default:
            return z;
    }
}

Matrix Layer::applySoftmaxJacobian(const Matrix& grad) const {
    Matrix localGrad(grad.getRows(), grad.getCols());

    for (int i = 0; i < grad.getRows(); i++) {
        double dot = 0.0;
        for (int j = 0; j < grad.getCols(); j++) {
            dot += grad(i, j) * A(i, j);
        }
        for (int j = 0; j < grad.getCols(); j++) {
            localGrad(i, j) = A(i, j) * (grad(i, j) - dot);
        }
    }

    return localGrad;
}

Matrix Layer::applyActivationGradient(const Matrix& grad) const {
    switch (act) {
        case SOFTMAX:
            return applySoftmaxJacobian(grad);
        case RELU:
            return grad * Z.map([](double x) { return x > 0.0 ? 1.0 : 0.0; });
        case LEAKY_RELU:
            return grad * Z.map([](double x) { return x > 0.0 ? 1.0 : 0.01; });
        case SIGMOID:
            return grad * A.map([](double a) { return a * (1.0 - a); });
        case TANH:
            return grad * A.map([](double a) { return 1.0 - a * a; });
        case NONE:
            return grad;
        default:
            return grad;
    }
}

Matrix Layer::forward(const Matrix& x, bool training){
    if (x.getCols() != W.getRows()) {
        throw std::invalid_argument(
            "Layer::forward input dimension mismatch: " +
            std::to_string(x.getRows()) + "x" + std::to_string(x.getCols()) +
            " | Nx" + std::to_string(W.getRows())
        );
    }

    X = x;
    Z = x.dot(W);
    for (int i = 0; i < Z.getRows(); i++) {
        for (int j = 0; j < Z.getCols(); j++) {
            Z(i, j) += b(0, j);
        }
    }

    A = applyActivation(Z);
    if (dropoutRate > 0.0) {
        if (training) {
            dropoutMask = Matrix(A.getRows(), A.getCols());
            double keepScale = 1.0 / (1.0 - dropoutRate);
            for (int i = 0; i < A.getRows(); i++) {
                for (int j = 0; j < A.getCols(); j++) {
                    double r = rand.uniform(0.0, 1.0);
                    dropoutMask(i, j) = (r >= dropoutRate) ? keepScale : 0.0;
                }
            }
            A = A * dropoutMask;
        }
        else {
            dropoutMask = Matrix();
        }
    }
    return A;
}

Matrix Layer::backward(const Matrix& grad, Optimizer& optimizer, int layerIndex, bool applyActivationDerivative){
    if (X.getRows() == 0 || Z.getRows() == 0) {
        throw std::logic_error("Layer::backward called before forward");
    }
    if (grad.getRows() != Z.getRows() || grad.getCols() != Z.getCols()) {
        throw std::invalid_argument(
            "Layer::backward gradient dimension mismatch: " +
            std::to_string(grad.getRows()) + "x" + std::to_string(grad.getCols()) +
            " | " + std::to_string(Z.getRows()) + "x" + std::to_string(Z.getCols())
        );
    }

    Matrix incomingGrad = grad;
    if (dropoutRate > 0.0) {
        if (dropoutMask.getRows() != grad.getRows() || dropoutMask.getCols() != grad.getCols()) {
            throw std::logic_error("Layer::backward missing dropout mask");
        }
        incomingGrad = incomingGrad * dropoutMask;
    }

    Matrix localGrad = applyActivationDerivative ? applyActivationGradient(incomingGrad) : incomingGrad;

    Matrix dW = X.transpose().dot(localGrad);
    Matrix db = localGrad.sum(0);
    Matrix dX = localGrad.dot(W.transpose());

    optimizer.update(layerIndex, W, b, dW, db);

    return dX;
}

void Layer::setWeights(const Matrix& w){
    if (w.getRows() <= 0 || w.getCols() <= 0) {
        throw std::invalid_argument("Layer::setWeights requires non-empty matrix");
    }
    if (w.getCols() != b.getCols()) {
        throw std::invalid_argument(
            "Layer::setWeights output dimension mismatch: expected " +
            std::to_string(b.getCols()) + " | " + std::to_string(w.getCols())
        );
    }
    W = w;
}

void Layer::setBias(const Matrix& bias){
    if (bias.getRows() != 1) {
        throw std::invalid_argument("Layer::setBias expects a row vector (1 x outputs)");
    }
    if (bias.getCols() != W.getCols()) {
        throw std::invalid_argument(
            "Layer::setBias output dimension mismatch: expected " +
            std::to_string(W.getCols()) + " | " + std::to_string(bias.getCols())
        );
    }
    b = bias;
}