#ifndef _MATRIX_H_
#define _MATRIX_H_

#include<vector>
#include <functional>

class Matrix{
    private:
        int rows, cols;
        std::vector<double> data;
    public:
        Matrix();
        Matrix(int rows, int columns);
        Matrix transpose() const;
        Matrix dot(const Matrix& B) const;
        Matrix map(std::function<double(double)> f) const;
        Matrix softmax() const;
        // geters
        int getCols() const;
        int getRows() const;
        std::vector<double> getData() const;
        // operators
        double& operator()(int i, int j);
        const double& operator()(int i, int j) const;
        Matrix operator+(const Matrix& other) const;
        Matrix operator-(const Matrix& other) const;
        Matrix operator*(double K) const;
        Matrix operator*(const Matrix& Mat) const;
        void print() const;
};

#endif