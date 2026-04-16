#include "Matrix.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

Matrix::Matrix(int rows, int cols): rows(0), cols(0) {
    if (rows < 0 || cols < 0) {
        throw std::invalid_argument(
            "Matrix::Matrix invalid dimensions: " +
            std::to_string(rows) + "x" + std::to_string(cols)
        );
    }

    this->rows = rows;
    this->cols = cols;
    data.assign((size_t)rows * (size_t)cols, 0.0);
}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.at(j, i) = at(i, j);
        }
    }

    return result;
}

Matrix Matrix::dot(const Matrix& mat) const {
    if(cols != mat.rows){
        throw std::invalid_argument("Matrix::dot dimension mismatch: " + std::to_string(rows) + "x" + std::to_string(cols) + " | " + std::to_string(mat.rows) + "x" + std::to_string(mat.cols));
    }

    Matrix result(rows, mat.cols);

    for (int i = 0; i < rows; i++) {
        for(int k = 0; k < cols; k++){
            double a = at(i, k);
            for (int j = 0; j < mat.cols; j++) {
                result.at(i, j) += a * mat.at(k, j);
            }
        }
    }

    return result;
}

Matrix Matrix::map(std::function<double(double)> func) const {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.at(i, j) = func(at(i, j));
        }
    }
    return result;
}

Matrix Matrix::softmax() const {
    Matrix result(rows, cols);
    if (rows == 0 || cols == 0) {
        return result;
    }

    for (int i = 0; i < rows; i++) {
        double maxVal = at(i, 0);
        for (int j = 1; j < cols; j++) {
            double value = at(i, j);
            if (value > maxVal) {
                maxVal = value;
            }
        }

        double sum = 0.0;
        for (int j = 0; j < cols; j++) {
            double shiftedExp = std::exp(at(i, j) - maxVal);
            result.at(i, j) = shiftedExp;
            sum += shiftedExp;
        }

        if (sum <= 0.0 || !std::isfinite(sum)) {
            throw std::runtime_error("Matrix::softmax invalid softmax normalization");
        }

        double invSum = 1.0 / sum;
        for (int j = 0; j < cols; j++) {
            result.at(i, j) *= invSum;
        }
    }
    return result;
}

Matrix Matrix::oneHot(int numClasses) {
    if (numClasses <= 0) {
        throw std::invalid_argument("Matrix::oneHot: numClasses must be > 0");
    }
    if (rows != 1) {
        throw std::invalid_argument("Matrix::oneHot: expected row vector (1 x n)");
    }
    
    Matrix result(numClasses, cols);
    for (int i = 0; i < cols; i++) {
        double rawLabel = at(0, i);
        double rounded = std::round(rawLabel);
        if (std::abs(rawLabel - rounded) > 1e-9) {
            throw std::invalid_argument(
                "Matrix::oneHot: label must be an integer, got " + std::to_string(rawLabel)
            );
        }

        int label = (int)rounded;
        if (label < 0 || label >= numClasses) {
            throw std::out_of_range("Matrix::oneHot: label " + std::to_string(label) + " out of range");
        }
        result.at(label, i) = 1.0;
    }
    return result;
}

Matrix Matrix::hstack(const Matrix& mat){
    if(this->rows != mat.rows){
        throw std::out_of_range("Matrix::hstack row mismatch: " + std::to_string(this->rows) + " | " + std::to_string(mat.rows));
    }
    if (this->cols == 0) return mat;
    if (mat.cols == 0) return *this;

    Matrix result(this->rows, this->cols + mat.cols);
    for(int i = 0; i < result.rows; i++){
        for (int j = 0; j < cols; j++) {
            result.at(i, j) = at(i, j);
        }
        for (int j = 0; j < mat.cols; j++) {
            result.at(i, cols + j) = mat.at(i, j);
        }
    }

    return result;
}

Matrix Matrix::vstack(const Matrix& mat){
    if(this->cols != mat.cols){
        throw std::out_of_range("Matrix::vstack column mismatch: " + std::to_string(this->cols) + " | " + std::to_string(mat.cols));
    }

    Matrix result(this->rows + mat.rows, this->cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.at(i, j) = at(i, j);
        }
    }

    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.at(rows + i, j) = mat.at(i, j);
        }
    }

    return result;
}

std::pair<Matrix, Matrix> Matrix::hsplit(int col){
    if(col < 0 || col > this->cols){
        throw std::out_of_range("Matrix::hsplit: col " + std::to_string(col) + " out of range (cols=" + std::to_string(this->cols) + ")");
    }
    Matrix X(this->rows, col);
    Matrix Y(this->rows, this->cols - col);

    for(int i=0; i<this->rows; i++){
        for (int j = 0; j < X.cols; j++) {
            X.at(i, j) = at(i, j);
        }
        for (int j = 0; j < Y.cols; j++) {
            Y.at(i, j) = at(i, X.cols + j);
        }
    }

    return {X, Y};
}

std::pair<Matrix, Matrix> Matrix::vsplit(int row){
    if(row < 0 || row > this->rows){
        throw std::out_of_range("Matrix::vsplit: row " + std::to_string(row) + " out of range (rows=" + std::to_string(this->rows) + ")");
    }
    Matrix X(row, this->cols);
    Matrix Y(this->rows - row, this->cols);

    for (int i = 0; i < X.rows; i++) {
        for (int j = 0; j < cols; j++) {
            X.at(i, j) = at(i, j);
        }
    }

    for (int i = 0; i < Y.rows; i++) {
        for (int j = 0; j < cols; j++) {
            Y.at(i, j) = at(row + i, j);
        }
    }

    return {X, Y};
}

Matrix Matrix::hslice(int colStart, int colEnd) const {
    if (colStart < 0 || colEnd > cols || colStart > colEnd) {
        throw std::out_of_range("Matrix::hslice: invalid range [" + std::to_string(colStart) + ", " + std::to_string(colEnd) + ")");
    }

    Matrix result(rows, colEnd - colStart);
    if (result.cols == 0) {
        return result;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < result.cols; j++) {
            result.at(i, j) = at(i, colStart + j);
        }
    }

    return result;
}

Matrix Matrix::vslice(int rowStart, int rowEnd) const {
    if (rowStart < 0 || rowEnd > rows || rowStart > rowEnd) {
        throw std::out_of_range("Matrix::vslice: invalid range [" + std::to_string(rowStart) + ", " + std::to_string(rowEnd) + ")");
    }

    Matrix result(rowEnd - rowStart, cols);
    if (result.getRows() == 0 || result.getCols() == 0) {
        return result;
    }

    for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.at(i, j) = at(rowStart + i, j);
        }
    }

    return result;
}

Matrix Matrix::shuffleRows(const std::vector<int>& rowOrder) const {
    if ((int)rowOrder.size() != rows) {
        throw std::invalid_argument(
            "Matrix::shuffleRows permutation size mismatch: expected " + std::to_string(rows) +
            " | " + std::to_string(rowOrder.size())
        );
    }

    Matrix result(rows, cols);
    std::vector<int> seen(rows, 0);

    for (int i = 0; i < rows; i++) {
        int srcRow = rowOrder[i];
        if (srcRow < 0 || srcRow >= rows) {
            throw std::out_of_range("Matrix::shuffleRows row index out of range: " + std::to_string(srcRow));
        }
        if (seen[srcRow] != 0) {
            throw std::invalid_argument("Matrix::shuffleRows duplicate row index: " + std::to_string(srcRow));
        }
        seen[srcRow] = 1;

        for (int j = 0; j < cols; j++) {
            result.at(i, j) = at(srcRow, j);
        }
    }

    return result;
}

Matrix Matrix::sum(int axis) const {
    if (axis == 0) {
        Matrix result(1, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.at(0, j) += at(i, j);
            }
        }
        return result;
    }

    if (axis == 1) {
        Matrix result(rows, 1);
        for (int i = 0; i < rows; i++) {
            double rowSum = 0.0;
            for (int j = 0; j < cols; j++) {
                rowSum += at(i, j);
            }
            result.at(i, 0) = rowSum;
        }
        return result;
    }

    throw std::invalid_argument("Matrix::sum axis must be 0 (columns) or 1 (rows)");
}

Matrix Matrix::mean(int axis) const {
    if (axis == 0) {
        if (rows <= 0) {
            throw std::invalid_argument("Matrix::mean axis 0 requires rows > 0");
        }
        return sum(0) * (1.0 / rows);
    }

    if (axis == 1) {
        if (cols <= 0) {
            throw std::invalid_argument("Matrix::mean axis 1 requires cols > 0");
        }
        return sum(1) * (1.0 / cols);
    }

    throw std::invalid_argument("Matrix::mean axis must be 0 (columns) or 1 (rows)");
}

Matrix Matrix::var(int axis) const {
    if (axis == 0) {
        if (rows <= 0) {
            throw std::invalid_argument("Matrix::var axis 0 requires rows > 0");
        }

        Matrix means = mean(0);
        Matrix result(1, cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double diff = at(i, j) - means.at(0, j);
                result.at(0, j) += diff * diff;
            }
        }

        return result * (1.0 / rows);
    }

    if (axis == 1) {
        if (cols <= 0) {
            throw std::invalid_argument("Matrix::var axis 1 requires cols > 0");
        }

        Matrix means = mean(1);
        Matrix result(rows, 1);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double diff = at(i, j) - means.at(i, 0);
                result.at(i, 0) += diff * diff;
            }
        }

        return result * (1.0 / cols);
    }

    throw std::invalid_argument("Matrix::var axis must be 0 (columns) or 1 (rows)");
}

Matrix Matrix::std(int axis) const {
    Matrix variance = var(axis);
    return variance.map([](double x) { return std::sqrt(x); });
}

Matrix Matrix::exp() const {
    Matrix result(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.at(i, j) = std::exp(at(i, j));
        }
    }

    return result;
}

Matrix Matrix::log() const {
    Matrix result(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double value = at(i, j);
            if (value <= 0.0) {
            throw std::domain_error(
                "Matrix::log domain error at (" + std::to_string(i) + ", " +
                std::to_string(j) + "): value must be > 0"
            );
        }

            result.at(i, j) = std::log(value);
        }
    }

    return result;
}

void Matrix::setRow(int row, const std::vector<double>& values) {
    if (row < 0 || row >= rows) {
        throw std::out_of_range("Matrix::setRow row index out of range");
    }
    if ((int)values.size() != cols) {
        throw std::invalid_argument("Matrix::setRow values size mismatch: expected " + std::to_string(cols) + " | " + std::to_string(values.size()));
    }

    for (int j = 0; j < cols; j++) {
        at(row, j) = values[j];
    }
}

void Matrix::setData(const std::vector<double>& d) {
    if (d.size() != (size_t)(rows * cols)) {
        throw std::invalid_argument(
            "Matrix::setData size mismatch: " + std::to_string(rows * cols) + " | " + std::to_string(d.size())
        );
    }
    data = d;
};

double& Matrix::operator()(int i, int j){
    if (i < 0 || i >= rows || j < 0 || j >= cols) {
        throw std::out_of_range("Matrix::operator() matrix index out of range");
    }
    return at(i, j);
}

const double& Matrix::operator()(int i, int j) const {
    if (i < 0 || i >= rows || j < 0 || j >= cols) {
        throw std::out_of_range("Matrix::operator() matrix index out of range");
    }
    return at(i, j);
}

Matrix Matrix::operator+(const Matrix& mat) const {
    if(cols != mat.cols || rows != mat.rows){
        throw std::invalid_argument(
            "Matrix::operator+ dimension mismatch: " + std::to_string(rows) + "x" + std::to_string(cols) +
            " | " + std::to_string(mat.rows) + "x" + std::to_string(mat.cols)
        );
    }

    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.at(i, j) = at(i, j) + mat.at(i, j);
        }
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& mat) const {
    if(cols != mat.cols || rows != mat.rows){
        throw std::invalid_argument(
            "Matrix::operator- dimension mismatch: " + std::to_string(rows) + "x" + std::to_string(cols) +
            " | " + std::to_string(mat.rows) + "x" + std::to_string(mat.cols)
        );
    }
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.at(i, j) = at(i, j) - mat.at(i, j);
        }
    }
    return result;
}

Matrix Matrix::operator*(double K) const {
    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.at(i, j) = K * at(i, j);
        }
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& mat) const {
    if(cols != mat.cols || rows != mat.rows){
        throw std::invalid_argument(
            "Matrix::operator* element-wise dimension mismatch: " + std::to_string(rows) + "x" + std::to_string(cols) +
            " | " + std::to_string(mat.rows) + "x" + std::to_string(mat.cols)
        );
    }

    Matrix result(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result.at(i, j) = at(i, j) * mat.at(i, j);
        }
    }
    return result;
}

std::ostream& operator<<(std::ostream& os,const Matrix& mat){
    for(int i=0; i<mat.rows; i++){
        for(int j=0; j<mat.cols; j++){
            os << mat.at(i, j);
            if (j < mat.cols - 1) {
                os << " ";
            }
        }
        os << "\n";
    }
    return os;
}