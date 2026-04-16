#include "Metrics.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

void Metrics::validateSameShape(const Matrix& Y, const Matrix& yPred, const std::string& methodName) {
    if (Y.getRows() <= 0 || Y.getCols() <= 0 || yPred.getRows() <= 0 || yPred.getCols() <= 0) {
        throw std::invalid_argument(methodName + " expects non-empty matrices");
    }

    if (Y.getRows() != yPred.getRows() || Y.getCols() != yPred.getCols()) {
        throw std::invalid_argument(
            methodName + " shape mismatch: " + std::to_string(Y.getRows()) + "x" + std::to_string(Y.getCols()) +
            " | " + std::to_string(yPred.getRows()) + "x" + std::to_string(yPred.getCols())
        );
    }
}

int Metrics::rowArgmax(const Matrix& mat, int row) {
    if (row < 0 || row >= mat.getRows()) {
        throw std::out_of_range("Metrics::rowArgmax row index out of range");
    }

    int cols = mat.getCols();
    if (cols <= 0) {
        throw std::invalid_argument("Metrics::rowArgmax expects cols > 0");
    }

    int bestIdx = 0;
    double bestVal = mat(row, 0);

    for (int j = 1; j < cols; ++j) {
        if (mat(row, j) > bestVal) {
            bestVal = mat(row, j);
            bestIdx = j;
        }
    }

    return bestIdx;
}

double Metrics::mse(const Matrix& Y, const Matrix& yPred) {
    validateSameShape(Y, yPred, "Metrics::mse");

    double errorSum = 0.0;
    for (int i = 0; i < Y.getRows(); ++i) {
        for (int j = 0; j < Y.getCols(); ++j) {
            double diff = yPred(i, j) - Y(i, j);
            errorSum += diff * diff;
        }
    }

    return errorSum / (Y.getRows() * Y.getCols());
}

double Metrics::mae(const Matrix& Y, const Matrix& yPred) {
    validateSameShape(Y, yPred, "Metrics::mae");

    double errorSum = 0.0;
    for (int i = 0; i < Y.getRows(); ++i) {
        for (int j = 0; j < Y.getCols(); ++j) {
            errorSum += std::abs(yPred(i, j) - Y(i, j));
        }
    }

    return errorSum / (Y.getRows() * Y.getCols());
}

double Metrics::r2Score(const Matrix& Y, const Matrix& yPred) {
    validateSameShape(Y, yPred, "Metrics::r2Score");

    Matrix yMeanByCol = Y.mean(0);

    double ssRes = 0.0;
    double ssTot = 0.0;

    for (int i = 0; i < Y.getRows(); ++i) {
        for (int j = 0; j < Y.getCols(); ++j) {
            double residual = Y(i, j) - yPred(i, j);
            double centered = Y(i, j) - yMeanByCol(0, j);
            ssRes += residual * residual;
            ssTot += centered * centered;
        }
    }

    if (ssTot <= 0.0) {
        return (ssRes <= 0.0) ? 1.0 : 0.0;
    }

    return 1.0 - (ssRes / ssTot);
}

double Metrics::accuracy(const Matrix& Y, const Matrix& yPred) {
    validateSameShape(Y, yPred, "Metrics::accuracy");

    int rows = Y.getRows();
    int cols = Y.getCols();
    int correct = 0;

    if (cols == 1) {
        for (int i = 0; i < rows; ++i) {
            int trueLabel = (Y(i, 0) >= 0.5) ? 1 : 0;
            int predLabel = (yPred(i, 0) >= 0.5) ? 1 : 0;
            if (trueLabel == predLabel) {
                correct++;
            }
        }
    }
    else {
        for (int i = 0; i < rows; ++i) {
            int trueLabel = rowArgmax(Y, i);
            int predLabel = rowArgmax(yPred, i);
            if (trueLabel == predLabel) {
                correct++;
            }
        }
    }

    return (double)correct / rows;
}

double Metrics::crossEntropy(const Matrix& Y, const Matrix& yPred, double epsilon) {
    validateSameShape(Y, yPred, "Metrics::crossEntropy");

    if (epsilon <= 0.0 || epsilon >= 1.0) {
        throw std::invalid_argument("Metrics::crossEntropy epsilon must be in (0, 1)");
    }

    int rows = Y.getRows();
    int cols = Y.getCols();
    double loss = 0.0;

    if (cols == 1) {
        for (int i = 0; i < rows; ++i) {
            double y = Y(i, 0);
            if (y < 0.0 || y > 1.0) {
                throw std::invalid_argument("Metrics::crossEntropy binary labels must be in [0, 1]");
            }
            if (!std::isfinite(yPred(i, 0))) {
                throw std::invalid_argument("Metrics::crossEntropy received non-finite prediction value");
            }

            double p = std::max(epsilon, std::min(1.0 - epsilon, yPred(i, 0)));
            loss += -(y * std::log(p) + (1.0 - y) * std::log(1.0 - p));
        }

        return loss / rows;
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double y = Y(i, j);
            if (y < 0.0) {
                throw std::invalid_argument("Metrics::crossEntropy labels must be >= 0 for multiclass");
            }
            if (!std::isfinite(yPred(i, j))) {
                throw std::invalid_argument("Metrics::crossEntropy received non-finite prediction value");
            }

            double p = std::max(epsilon, std::min(1.0 - epsilon, yPred(i, j)));
            loss += -(y * std::log(p));
        }
    }

    return loss / rows;
}
