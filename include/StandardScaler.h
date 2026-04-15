#ifndef _STANDARD_SCALER_H_
#define _STANDARD_SCALER_H_

#include <vector>
#include "Matrix.h"

class StandardScaler {
    private:
        std::vector<double> mean;
        std::vector<double> scale;
        bool fitted;

    public:
        StandardScaler() : fitted(false) {}
        void fit(const Matrix& X);
        Matrix transform(const Matrix& X) const;
        Matrix fitTransform(const Matrix& X);
        Matrix inverseTransform(const Matrix& X) const;
        bool isFitted() const { return fitted; }
        const std::vector<double>& getMean() const { return mean; }
        const std::vector<double>& getScale() const { return scale; }
};

#endif