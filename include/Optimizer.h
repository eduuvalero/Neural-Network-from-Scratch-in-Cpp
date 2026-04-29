#ifndef _OPTIMIZER_H_
#define _OPTIMIZER_H_

#include <vector>

#include "Matrix.h"

class Optimizer {
    public:
        virtual ~Optimizer() {}
        virtual void reset() = 0;
        virtual void step() = 0;
        virtual void update(int layerIndex, Matrix& W, Matrix& b, const Matrix& dW, const Matrix& db) = 0;
};

class SGD : public Optimizer {
    private:
        double lr;
    public:
        SGD(double lr = 0.01);
        void setLearningRate(double lr);
        double getLearningRate() const { return lr; }
        void reset() override {}
        void step() override {}
        void update(int layerIndex, Matrix& W, Matrix& b, const Matrix& dW, const Matrix& db) override;
};

class Adam : public Optimizer {
    private:
        double lr;
        double beta1;
        double beta2;
        double eps;
        int t;
        std::vector<Matrix> mW;
        std::vector<Matrix> vW;
        std::vector<Matrix> mB;
        std::vector<Matrix> vB;
        void ensureState(int layerIndex, const Matrix& W, const Matrix& b);
    public:
        Adam(double lr = 0.001, double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8);
        void setLearningRate(double lr);
        double getLearningRate() const { return lr; }
        double getBeta1() const { return beta1; }
        double getBeta2() const { return beta2; }
        double getEps() const { return eps; }
        int getT() const { return t; }
        const std::vector<Matrix>& getMW() const { return mW; }
        const std::vector<Matrix>& getVW() const { return vW; }
        const std::vector<Matrix>& getMB() const { return mB; }
        const std::vector<Matrix>& getVB() const { return vB; }
        void loadState(int t, const std::vector<Matrix>& mW, const std::vector<Matrix>& vW, const std::vector<Matrix>& mB, const std::vector<Matrix>& vB);
        void reset() override;
        void step() override;
        void update(int layerIndex, Matrix& W, Matrix& b, const Matrix& dW, const Matrix& db) override;
};

#endif
