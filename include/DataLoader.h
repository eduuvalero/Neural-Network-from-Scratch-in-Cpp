#ifndef _DATA_LOADER_H_
#define _DATA_LOADER_H_

#include <string>
#include <vector>
#include "Matrix.h"

class DataLoader{
    private:
        static std::vector<double> parseRow(const std::string& line);
        static Matrix readCSV(const std::string& path);
        static int normalizeLabelCol(int labelCol, int numCols);

    public:
        static std::pair<Matrix, Matrix> loadDataset(const std::string& path, int labelCol = -1, int numClasses = 1);
        static Matrix loadX(const std::string& path) { return readCSV(path); }
        static Matrix loadY(const std::string& path, int labelCol = -1, int numClasses = 1);
        static void saveCSV(const std::string& path, const Matrix& data);
};

#endif