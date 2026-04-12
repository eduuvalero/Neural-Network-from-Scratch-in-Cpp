#ifndef _DATA_LOADER_H_
#define _DATA_LOADER_H_

#include <string>
#include "Matrix.h"

class DataLoader{
    public:
        static void loadCSV(const std::string& path, Matrix& X, Matrix& Y, int labelCol = 0, int numClasses = 1);
        static Matrix loadX(const std::string& path);
        static Matrix loadY(const std::string& path, int labelCol = 0, int numClasses = 1);
        void saveCSV(const std::string& path, const Matrix& data);
};

#endif