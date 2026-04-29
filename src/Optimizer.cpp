#include "Optimizer.h"

#include <cmath>
#include <stdexcept>

SGD::SGD(double lr): lr(lr) {
    if (lr <= 0.0) {
        throw std::invalid_argument("SGD::SGD learning rate must be > 0");
    }
}

void SGD::setLearningRate(double lr) {
    if (lr <= 0.0) {
        throw std::invalid_argument("SGD::setLearningRate learning rate must be > 0");
    }
    this->lr = lr;
}

void SGD::update(int layerIndex, Matrix& W, Matrix& b, const Matrix& dW, const Matrix& db) {
    (void)layerIndex;
    if (dW.getRows() != W.getRows() || dW.getCols() != W.getCols()) {
        throw std::invalid_argument("SGD::update weight gradient dimension mismatch");
    }
    if (db.getRows() != b.getRows() || db.getCols() != b.getCols()) {
        throw std::invalid_argument("SGD::update bias gradient dimension mismatch");
    }

    W = W - (dW * lr);
    b = b - (db * lr);
}

Adam::Adam(double lr, double beta1, double beta2, double eps):
    lr(lr),
    beta1(beta1),
    beta2(beta2),
    eps(eps),
    t(0) {
    if (lr <= 0.0) {
        throw std::invalid_argument("Adam::Adam learning rate must be > 0");
    }
    if (beta1 <= 0.0 || beta1 >= 1.0) {
        throw std::invalid_argument("Adam::Adam beta1 must be in (0, 1)");
    }
    if (beta2 <= 0.0 || beta2 >= 1.0) {
        throw std::invalid_argument("Adam::Adam beta2 must be in (0, 1)");
    }
    if (eps <= 0.0) {
        throw std::invalid_argument("Adam::Adam eps must be > 0");
    }
}

void Adam::setLearningRate(double lr) {
    if (lr <= 0.0) {
        throw std::invalid_argument("Adam::setLearningRate learning rate must be > 0");
    }
    this->lr = lr;
}

void Adam::loadState(int t, const std::vector<Matrix>& mW, const std::vector<Matrix>& vW, const std::vector<Matrix>& mB, const std::vector<Matrix>& vB) {
    if (t < 0) {
        throw std::invalid_argument("Adam::loadState t must be >= 0");
    }
    if (mW.size() != vW.size()) {
        throw std::invalid_argument("Adam::loadState weight state size mismatch");
    }
    if (mB.size() != vB.size()) {
        throw std::invalid_argument("Adam::loadState bias state size mismatch");
    }
    if (mW.size() != mB.size()) {
        throw std::invalid_argument("Adam::loadState layer count mismatch between weights and bias state");
    }

    for (size_t i = 0; i < mW.size(); i++) {
        if (mW[i].getRows() != vW[i].getRows() || mW[i].getCols() != vW[i].getCols()) {
            throw std::invalid_argument("Adam::loadState weight matrix dimension mismatch");
        }
    }
    for (size_t i = 0; i < mB.size(); i++) {
        if (mB[i].getRows() != vB[i].getRows() || mB[i].getCols() != vB[i].getCols()) {
            throw std::invalid_argument("Adam::loadState bias matrix dimension mismatch");
        }
    }

    this->t = t;
    this->mW = mW;
    this->vW = vW;
    this->mB = mB;
    this->vB = vB;
}

void Adam::reset() {
    t = 0;
    mW.clear();
    vW.clear();
    mB.clear();
    vB.clear();
}

void Adam::step() {
    t++;
}

void Adam::ensureState(int layerIndex, const Matrix& W, const Matrix& b) {
    if (layerIndex < 0) {
        throw std::invalid_argument("Adam::ensureState layer index must be >= 0");
    }

    int required = layerIndex + 1;
    if ((int)mW.size() < required) {
        mW.resize(required);
        vW.resize(required);
        mB.resize(required);
        vB.resize(required);
    }

    if (mW[layerIndex].getRows() == 0 && mW[layerIndex].getCols() == 0) {
        mW[layerIndex] = Matrix(W.getRows(), W.getCols());
        vW[layerIndex] = Matrix(W.getRows(), W.getCols());
        mB[layerIndex] = Matrix(b.getRows(), b.getCols());
        vB[layerIndex] = Matrix(b.getRows(), b.getCols());
        return;
    }

    if (mW[layerIndex].getRows() != W.getRows() || mW[layerIndex].getCols() != W.getCols()) {
        throw std::invalid_argument("Adam::ensureState weight state dimension mismatch");
    }
    if (mB[layerIndex].getRows() != b.getRows() || mB[layerIndex].getCols() != b.getCols()) {
        throw std::invalid_argument("Adam::ensureState bias state dimension mismatch");
    }
}

void Adam::update(int layerIndex, Matrix& W, Matrix& b, const Matrix& dW, const Matrix& db) {
    if (t <= 0) {
        throw std::logic_error("Adam::update step() must be called before update()");
    }
    if (dW.getRows() != W.getRows() || dW.getCols() != W.getCols()) {
        throw std::invalid_argument("Adam::update weight gradient dimension mismatch");
    }
    if (db.getRows() != b.getRows() || db.getCols() != b.getCols()) {
        throw std::invalid_argument("Adam::update bias gradient dimension mismatch");
    }

    ensureState(layerIndex, W, b);

    mW[layerIndex] = (mW[layerIndex] * beta1) + (dW * (1.0 - beta1));
    vW[layerIndex] = (vW[layerIndex] * beta2) + ((dW * dW) * (1.0 - beta2));
    mB[layerIndex] = (mB[layerIndex] * beta1) + (db * (1.0 - beta1));
    vB[layerIndex] = (vB[layerIndex] * beta2) + ((db * db) * (1.0 - beta2));

    double beta1Corr = 1.0 - std::pow(beta1, t);
    double beta2Corr = 1.0 - std::pow(beta2, t);

    Matrix mHatW = mW[layerIndex] * (1.0 / beta1Corr);
    Matrix vHatW = vW[layerIndex] * (1.0 / beta2Corr);
    Matrix mHatB = mB[layerIndex] * (1.0 / beta1Corr);
    Matrix vHatB = vB[layerIndex] * (1.0 / beta2Corr);

    double epsilon = eps;
    Matrix denomW = vHatW.map([epsilon](double x) { return 1.0 / (std::sqrt(x) + epsilon); });
    Matrix denomB = vHatB.map([epsilon](double x) { return 1.0 / (std::sqrt(x) + epsilon); });

    Matrix stepW = mHatW * denomW;
    Matrix stepB = mHatB * denomB;

    W = W - (stepW * lr);
    b = b - (stepB * lr);
}
