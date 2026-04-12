#include "Matrix.h"

#include <stdexcept>
#include<iostream>
#include<cmath>

Matrix::Matrix() : rows(0), cols(0) {}

Matrix::Matrix(int rows, int cols): rows(rows), cols(cols), data(rows * cols, 0.0) {}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows); 
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result(j, i) = (*this)(i,j);
        }
    }
    return result;
}

Matrix Matrix::dot(const Matrix& mat) const {
    if(cols != mat.rows){
        throw std::invalid_argument("Incompatible dimensions");
    }

    Matrix result(rows, mat.cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            for(int k = 0; k < cols; k++){
                result(i,j) += (*this)(i,k) * mat(k,j);
            }
        }
    }
    return result;
}

Matrix Matrix::map(std::function<double(double)> func) const {
    Matrix result(rows, cols); 
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result(i,j) = func( (*this)(i,j) );
        }
    }
    return result;
}

Matrix Matrix::softmax() const {
    Matrix result(rows, cols);
    if (rows == 0 || cols == 0) {
        return result;
    }

    for (int j = 0; j < cols; j++) {
        double maxVal = (*this)(0, j);
        for (int i = 1; i < rows; i++) {
            if ((*this)(i, j) > maxVal) {
                maxVal = (*this)(i, j);
            }
        }

        double sum = 0.0;
        for (int i = 0; i < rows; i++) {
            sum += std::exp((*this)(i, j) - maxVal);
        }

        if (sum <= 0.0 || !std::isfinite(sum)) {
            throw std::runtime_error("Invalid softmax normalization");
        }

        for (int i = 0; i < rows; i++) {
            result(i, j) = std::exp((*this)(i, j) - maxVal) / sum;
        }
    }
    return result;
}

int Matrix::getCols() const { return cols; };

int Matrix::getRows() const { return rows; };

std::vector<double> Matrix::getData() const { return data; };

double& Matrix::operator()(int i, int j){
#ifndef NDEBUG
    if (i < 0 || i >= rows || j < 0 || j >= cols) {
        throw std::out_of_range("Matrix index out of range");
    }
#endif
    return data[i * cols + j];
}

const double& Matrix::operator()(int i, int j) const {
#ifndef NDEBUG
    if (i < 0 || i >= rows || j < 0 || j >= cols) {
        throw std::out_of_range("Matrix index out of range");
    }
#endif
    return data[i * cols + j];
}

Matrix Matrix::operator+(const Matrix& mat) const {
    if(cols != mat.cols || rows != mat.rows){
        throw std::invalid_argument("Incompatible dimensions");
    }
    Matrix result(rows, cols); 
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result(i,j) = (*this)(i,j) + mat(i,j);
        }
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& mat) const {
    if(cols != mat.cols || rows != mat.rows){
        throw std::invalid_argument("Incompatible dimensions");
    }
    Matrix result(rows, cols); 
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result(i,j) = (*this)(i,j) - mat(i,j);
        }
    }
    return result;
}

Matrix Matrix::operator*(double K) const {
    Matrix result(rows, cols); 
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result(i,j) = K * (*this)(i,j);
        }
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& mat) const {
    if(cols != mat.cols || rows != mat.rows){
        throw std::invalid_argument("Incompatible dimensions");
    }
    Matrix result(rows, cols); 
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result(i,j) = (*this)(i,j) * mat(i,j);
        }
    }
    return result;
}

void Matrix::print() const {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << (*this)(i, j) << " ";
        }
    }
    std::cout << "\n";
}