#include "Layer.h"

#include <cmath>
#include <random>
#include <string>
#include <stdexcept>

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

Layer::Layer(int inputs, int outputs, Activation act): W(inputs, outputs), b(1, outputs), act(act){
    if (inputs <= 0 || outputs <= 0) {
        throw std::invalid_argument(
            "Layer::Layer invalid dimensions: " +
            std::to_string(inputs) + "x" + std::to_string(outputs)
        );
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 0.1);

    for (int i = 0; i < inputs; i++) {
        for (int j = 0; j < outputs; j++) {
            W(i, j) = dist(gen);
        }
    }
}

Matrix Layer::forward(const Matrix& x){
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
    return A;
}

Matrix Layer::backward(const Matrix& grad, double lr, bool applyActivationDerivative){
    if (lr <= 0.0) {
        throw std::invalid_argument("Layer::backward learning rate must be > 0");
    }
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

    Matrix localGrad = applyActivationDerivative ? applyActivationGradient(grad) : grad;

    Matrix dW = X.transpose().dot(localGrad);
    Matrix db = localGrad.sum(0);
    Matrix dX = localGrad.dot(W.transpose());

    W = W - dW * lr;
    b = b - db * lr;

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