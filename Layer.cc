#include "Layer.h"

#include <random>
#include <cmath>

Layer::Layer(int inputs, int outputs, Activation act): W(outputs, inputs), b(outputs, 1), act(act){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 0.1);

    for (int i = 0; i < outputs; i++)
        for (int j = 0; j < inputs; j++)
            W(i, j) = dist(gen);
}

Matrix Layer::forward(const Matrix& x){
    X = x;
    Z = W.dot(x) + b;
    switch (act){
        case RELU: 
            return Z.map([](double x) { return x > 0 ? x : 0; });
        case LEAKY_RELU:
            return Z.map([](double x) { return x > 0 ? x : 0.01 * x; });
        case SIGMOID:
            return Z.map([](double x) { return 1.0 / (1.0 + exp(-x)); });
        case TANH:
            return Z.map([](double x) { return tanh(x); });
        case SOFTMAX:
            return Z.softmax();
        case NONE:
            return Z;
        default:
            return Z;
    }
}

Matrix Layer::backward(const Matrix& grad, double lr, bool applyActivationDerivative){
    Matrix act_grad;
    if (applyActivationDerivative) {
        switch (act){
            case RELU: 
                act_grad = Z.map([](double x) { return x > 0 ? 1.0 : 0.0; });
                break;
            case LEAKY_RELU:
                act_grad = Z.map([](double x) { return x > 0 ? 1.0 : 0.01; });
                break;
            case SIGMOID:
                act_grad = Z.map([](double x) { return ((1.0 / (1.0 + exp(-x))) * (1 - (1.0 / (1.0 + exp(-x))))); });
                break;
            case TANH:
                act_grad = Z.map([](double x) { return 1 - tanh(x) * tanh(x); });
                break;
            case SOFTMAX:
                act_grad = Z.map([](double x) { return 1.0; });
                break;
            case NONE:
                act_grad = Z.map([](double x) { return 1.0; });
                break;
            default:
                act_grad = Z.map([](double x) { return 1.0; });
        }
    } else {
        act_grad = Z.map([](double) { return 1.0; });
    }
    Matrix localGrad = grad * act_grad;
    Matrix dW = localGrad.dot(X.transpose());
    Matrix db = localGrad;
    Matrix dX = W.transpose().dot(localGrad);

    W = W - dW * lr;
    b = b - db * lr;

    return dX;
}

Activation Layer::getAct() const { return act; }
Matrix Layer::getWeights() const{ return W; }
Matrix Layer::getBias() const{ return b; }

void Layer::setWeights(const Matrix& w){ W = w;  }
void Layer::setBias(const Matrix& b){ this->b=b; }