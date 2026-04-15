#ifndef _METRICS_H_
#define _METRICS_H_

#include <cmath>
#include <string>

#include "Matrix.h"

class Metrics {
    private:
        static void validateSameShape(const Matrix& yTrue, const Matrix& yPred, const std::string& methodName);
        static int rowArgmax(const Matrix& mat, int row);

    public:
        static double mse(const Matrix& yTrue, const Matrix& yPred);
        static double mae(const Matrix& yTrue, const Matrix& yPred);
        static double rmse(const Matrix& yTrue, const Matrix& yPred) { return std::sqrt(mse(yTrue, yPred)); }
        static double r2Score(const Matrix& yTrue, const Matrix& yPred);
        static double accuracy(const Matrix& yTrue, const Matrix& yPred);
        static double crossEntropy(const Matrix& yTrue, const Matrix& yPred, double epsilon = 1e-12);
};

#endif
