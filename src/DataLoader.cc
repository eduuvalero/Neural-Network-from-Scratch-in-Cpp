#include "DataLoader.h"

#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <algorithm>

std::vector<double> DataLoader::parseRow(const std::string& line) {
    std::stringstream ss(line);
    std::string val;
    std::vector<double> row;
    row.reserve(std::count(line.begin(), line.end(), ',') + 1);

    while (std::getline(ss, val, ',')) {
        try {
            row.push_back(std::stod(val));
        } catch (...) {
            throw std::invalid_argument("Invalid value: '" + val + "'");
        }
    }
    return row;
}

Matrix DataLoader::readCSV(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::invalid_argument("DataLoader can't open file: " + path);
    }

    std::string line;
    std::vector<double> flatData;
    int lineNo = 0;
    int numCols = -1;
    int numRows = 0;

    while (std::getline(file, line)) {
        lineNo++;
        if (line.empty()) {
            continue;
        }
        try {
            std::vector<double> row = parseRow(line);
            if (numCols == -1) {
                numCols = row.size();
            }
            if ((int)row.size() != numCols) {
                std::cerr << "Warning: skipping row " << lineNo << " (inconsistent columns)\n";
                continue;
            }

            flatData.insert(flatData.end(), row.begin(), row.end());
            numRows++;
        } catch (...) {
            std::cerr << "Warning: skipping row " << lineNo << " (invalid value)\n";
        }
    }

    if (numRows == 0) {
        throw std::invalid_argument("No valid data in: " + path);
    }

    Matrix data(numRows, numCols);
    data.setData(flatData);

    return data;
}

int DataLoader::normalizeLabelCol(int labelCol, int numCols) {
    if (numCols <= 0) {
        throw std::invalid_argument("DataLoader::normalizeLabelCol invalid number of columns: " + std::to_string(numCols));
    }
    else {
        if (labelCol < 0) {
            labelCol = numCols + labelCol;
        }

        if (labelCol < 0 || labelCol>= numCols) {
            throw std::invalid_argument("DataLoader::normalizeLabelCol labelCol out of range: " + std::to_string(labelCol));
        }
    }

    return labelCol;
}

std::pair<Matrix, Matrix> DataLoader::loadDataset(const std::string& path, int labelCol, int numClasses) {
    if (numClasses <= 0) {
        throw std::invalid_argument("DataLoader::loadDataset: numClasses must be >= 1");
    }

    Matrix A = readCSV(path);
    int realLabelCol = normalizeLabelCol(labelCol, A.getCols());

    if (A.getCols() < 2) {
        throw std::invalid_argument("DataLoader::loadDataset: dataset must have at least 2 columns");
    }

    Matrix left = (realLabelCol > 0) ? A.hslice(0, realLabelCol) : Matrix(A.getRows(), 0);
    Matrix Y = A.hslice(realLabelCol, realLabelCol + 1);
    Matrix right = (realLabelCol < A.getCols() - 1) ? A.hslice(realLabelCol + 1, A.getCols()) : Matrix(A.getRows(), 0);
    Matrix X = left.hstack(right);

    if (numClasses > 1) {
        Y = Y.transpose().oneHot(numClasses).transpose();
    }

    return {X, Y};
}

Matrix DataLoader::loadY(const std::string& path, int labelCol, int numClasses) {
    if (numClasses <= 0) {
        throw std::invalid_argument("DataLoader::loadDataset: numClasses must be >= 1");
    }

    Matrix A = readCSV(path);
    int realLabelCol = normalizeLabelCol(labelCol, A.getCols());
    Matrix Y = A.hslice(realLabelCol, realLabelCol + 1);

    if (numClasses > 1) {
        Y = Y.transpose().oneHot(numClasses).transpose();
    }

    return Y;
}

void DataLoader::saveCSV(const std::string& path, const Matrix& data) {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::invalid_argument("DataLoader::saveCSV can't open file: " + path);
    }

    for (int i = 0; i < data.getRows(); i++) {
        for (int j = 0; j < data.getCols(); j++) {
            file << data(i, j);
            if (j < data.getCols() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }
}