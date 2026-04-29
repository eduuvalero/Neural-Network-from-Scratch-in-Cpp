#ifndef _LINEAR_REGRESSION_H_
#define _LINEAR_REGRESSION_H_

#include <cstddef>
#include <fstream>
#include <string>

#include "Matrix.h"
#include "Optimizer.h"

class LinearRegression{
    private:
        struct CheckpointMetrics {
            double mse = 0.0;
            double mae = 0.0;
            double rmse = 0.0;
            double r2 = 0.0;
        };

        inline static constexpr char kModelMagic[4] = {'L', 'R', 'M', '2'};
        inline static constexpr int kModelVersion = 2;
        inline static constexpr char kCheckpointMagic[4] = {'L', 'R', 'C', 'P'};
        inline static constexpr int kCheckpointVersion = 1;

        Matrix weights;
        double bias;
        bool hasCompileConfig;
        int compiledBatchSize;
        int compiledShuffleSeed;
        bool compiledLogMetrics;
        int compiledMetricsEvery;
        Optimizer* compiledOptimizer;
        SGD compiledSgd;
        Adam compiledAdam;
        Matrix forwardLinear(const Matrix& X) const;
        double sumColumnVector(const Matrix& vec) const;
        void train(const Matrix& X, const Matrix& Y, int epochs, Optimizer& optimizer, int batchSize, int shuffleSeed, bool logMetrics, int metricsEvery, int checkpointEvery, const std::string& checkpointDir);
        static void writeOrThrow(std::ofstream& file, const void* data, std::size_t size, const std::string& context);
        static void readOrThrow(std::ifstream& file, void* data, std::size_t size, const std::string& context);
        static void writeMatrixData(std::ofstream& file, const Matrix& mat, const std::string& context);
        static void readMatrixData(std::ifstream& file, Matrix& mat, const std::string& context);
        void saveCheckpoint(const std::string& checkpointDir, int epoch, int totalEpochs, int batchSize, int effectiveBatchSize, int shuffleSeed, bool logMetrics, int metricsEvery, const CheckpointMetrics& metrics);

    public:
        LinearRegression(int inputs);
        void compile(double lr = 0.01, int batchSize = 0, int shuffleSeed = -1, bool logMetrics = false, int metricsEvery = 1);
        void compile(Optimizer& optimizer, int batchSize = 0, int shuffleSeed = -1, bool logMetrics = false, int metricsEvery = 1);
        void fit(const Matrix& X, const Matrix& Y, int epochs);
        void fit(const Matrix& X, const Matrix& Y, int epochs, int checkpointEvery, const std::string& checkpointDir = "checkpoints");
        Matrix predict(const Matrix& x);
        void save(const std::string& path);
        void load(const std::string& path);
};

#endif