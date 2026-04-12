#include "DataLoader.h"

#include <fstream>
#include <sstream>
#include <vector>

void DataLoader::loadCSV(const std::string& path, Matrix& X, Matrix& Y, int labelCol, int numClasses) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::invalid_argument("Can't open file");

    std::string line;
    std::vector<std::vector<double>> data;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string val;
        std::vector<double> row;
        while (std::getline(ss, val, ','))
            row.push_back(std::stod(val));
        data.push_back(row);
    }

    int numSamples = data.size();
    int numCols = data[0].size();
    int numFeatures = numCols - 1;

    if (labelCol < 0)
        labelCol = numCols + labelCol;

    X = Matrix(numFeatures, numSamples);
    Y = Matrix(numClasses, numSamples);

    for (int i = 0; i < numSamples; i++) {
        int xCol = 0;
        for (int j = 0; j < numCols; j++) {
            if (j == labelCol)
                if (numClasses == 1)
                    Y(0, i) = data[i][j];
                else {
                    int label = (int)data[i][j];
                    Y(label, i) = 1.0;
                }
            else
                X(xCol++, i) = data[i][j];
        }
    }
}

Matrix DataLoader::loadX(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::invalid_argument("Can't open file");

    std::string line;
    std::vector<std::vector<double>> data;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string val;
        std::vector<double> row;
        while (std::getline(ss, val, ','))
            row.push_back(std::stod(val));
        data.push_back(row);
    }

    int numSamples = data.size();
    int numFeatures = data[0].size();
    Matrix X(numFeatures, numSamples);
    for (int i = 0; i < numSamples; i++)
        for (int j = 0; j < numFeatures; j++)
            X(j, i) = data[i][j];
    return X;
}

Matrix DataLoader::loadY(const std::string& path, int labelCol, int numClasses) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::invalid_argument("Can't open file");

    std::string line;
    std::vector<double> labels;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string val;
        std::vector<double> row;
        while (std::getline(ss, val, ','))
            row.push_back(std::stod(val));
    
        int col = (labelCol < 0) ? (int)row.size() + labelCol : labelCol;
        labels.push_back(row[col]);
    }
    
    int numSamples = labels.size();
    Matrix Y(numClasses, numSamples);
    for (int i = 0; i < numSamples; i++) {
        if (numClasses == 1)
            Y(0, i) = labels[i];
        else {
            int label = (int)labels[i];
            Y(label, i) = 1.0;
        }
    }
    return Y;
}

void DataLoader::saveCSV(const std::string& path, const Matrix& data) {
    std::ofstream file(path);
    if (!file.is_open())
        throw std::invalid_argument("Can't open file");

    for (int i = 0; i < data.getRows(); i++) {
        for (int j = 0; j < data.getCols(); j++) {
            file << data(i, j);
            if (j < data.getCols() - 1)
                file << ",";
        }
        file << "\n";
    }
}