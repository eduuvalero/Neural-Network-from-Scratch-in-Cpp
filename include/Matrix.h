#ifndef _MATRIX_H_
#define _MATRIX_H_

#include<iostream>
#include <vector>
#include <functional>

class Matrix{
    friend std::ostream& operator<<(std::ostream& os, const Matrix& mat);
    private:
        int rows, cols;
        std::vector<double> data;
        double& at(int i, int j) { return data[(size_t)i * (size_t)cols + (size_t)j]; }
        const double& at(int i, int j) const { return data[(size_t)i * (size_t)cols + (size_t)j]; }
    public:
        Matrix() : rows(0), cols(0) {}
        Matrix(int rows, int columns);
        // linear algebra
        Matrix transpose() const;
        Matrix dot(const Matrix& B) const;
        Matrix map(std::function<double(double)> f) const;
        Matrix softmax() const;
        Matrix oneHot(int numClasses);
        // Matrix operations
        Matrix hstack(const Matrix& mat);
        Matrix vstack(const Matrix& mat);
        std::pair<Matrix, Matrix> hsplit(int col);
        std::pair<Matrix, Matrix> vsplit(int row);
        Matrix hslice(int colStart, int colEnd) const;
        Matrix vslice(int rowStart, int rowEnd) const;
        Matrix shuffleRows(const std::vector<int>& rowOrder) const;
        Matrix sum(int axis) const;
        Matrix mean(int axis) const;
        Matrix var(int axis) const;
        Matrix std(int axis) const;
        Matrix exp() const;
        Matrix log() const;
        // geters
        int getCols() const { return cols; }
        int getRows() const { return rows; }
        const std::vector<double>& getData() const { return data; }
        // setters
        void setData(const std::vector<double>& d);
        void setRow(int row, const std::vector<double>& values);
        // operators
        double& operator()(int i, int j);
        const double& operator()(int i, int j) const;
        Matrix operator+(const Matrix& other) const;
        Matrix operator-(const Matrix& other) const;
        Matrix operator*(double K) const;
        Matrix operator*(const Matrix& Mat) const;
};

#endif