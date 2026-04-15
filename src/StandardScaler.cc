#include "StandardScaler.h"

#include <cmath>
#include <stdexcept>
#include <string>

void StandardScaler::fit(const Matrix& X) {
    if (X.getRows() <= 0 || X.getCols() <= 0) {
        throw std::invalid_argument("StandardScaler::fit matrix must be non-empty");
    }

    int cols = X.getCols();

    mean.assign(cols, 0.0);
    scale.assign(cols, 1.0);

    Matrix meanMatrix = X.mean(0);
    Matrix stdMatrix = X.std(0);

    for (int j = 0; j < cols; ++j) {
        mean[j] = meanMatrix(0, j);
        double stdDev = stdMatrix(0, j);

        if (!std::isfinite(mean[j]) || !std::isfinite(stdDev)) {
            throw std::runtime_error("StandardScaler::fit encountered non-finite statistics");
        }

        scale[j] = (std::abs(stdDev) > 1e-12) ? stdDev : 1.0;
    }

    fitted = true;
}

Matrix StandardScaler::transform(const Matrix& X) const {
    if (!fitted) {
        throw std::runtime_error("StandardScaler::transform called before fit");
    }
    if (X.getCols() != (int)mean.size()) {
        throw std::invalid_argument(
            "StandardScaler::transform feature mismatch: expected " + std::to_string(mean.size()) +
            ", got " + std::to_string(X.getCols())
        );
    }

    Matrix result(X.getRows(), X.getCols());
    for (int i = 0; i < X.getRows(); ++i) {
        for (int j = 0; j < X.getCols(); ++j) {
            result(i, j) = (X(i, j) - mean[j]) / scale[j];
        }
    }

    return result;
}

Matrix StandardScaler::fitTransform(const Matrix& X) {
    fit(X);
    return transform(X);
}

Matrix StandardScaler::inverseTransform(const Matrix& X) const {
    if (!fitted) {
        throw std::runtime_error("StandardScaler::inverseTransform called before fit");
    }
    if (X.getCols() != (int)mean.size()) {
        throw std::invalid_argument("StandardScaler::inverseTransform feature mismatch: expected " + std::to_string(mean.size()) +  " | " + std::to_string(X.getCols()));
    }

    Matrix result(X.getRows(), X.getCols());
    for (int i = 0; i < X.getRows(); ++i) {
        for (int j = 0; j < X.getCols(); ++j) {
            result(i, j) = X(i, j) * scale[j] + mean[j];
        }
    }

    return result;
}

