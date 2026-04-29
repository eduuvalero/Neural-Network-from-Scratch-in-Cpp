#ifndef _NEURAL_NETWORK_H_
#define _NEURAL_NETWORK_H_

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>
#include"Matrix.h"
#include"Layer.h"
#include"Optimizer.h"

enum Loss { MSE, CROSS_ENTROPY, AUTO_LOSS };

struct Dense {
    int outputs;
    Activation act;
    Inicialization init;
    double dropoutRate;

    Dense(int outputs, Activation act = NONE, Inicialization init = AUTO, double dropoutRate = 0.0): outputs(outputs), act(act), init(init), dropoutRate(dropoutRate) {}
};

class NeuralNetwork{
    private:
        struct CheckpointMetrics {
            int type = 0;
            double m1 = 0.0;
            double m2 = 0.0;
            double m3 = 0.0;
            double m4 = 0.0;
        };

        inline static constexpr char kModelMagic[4] = {'N', 'N', 'M', '2'};
        inline static constexpr int kModelVersion = 2;
        inline static constexpr char kCheckpointMagic[4] = {'N', 'N', 'C', 'P'};
        inline static constexpr int kCheckpointVersion = 1;

        std::vector<Layer> layers;
        int inputFeatures;
        bool hasCompileConfig;
        Loss compiledLoss;
        int compiledBatchSize;
        int compiledShuffleSeed;
        bool compiledLogMetrics;
        int compiledMetricsEvery;
        Optimizer* compiledOptimizer;
        SGD compiledSgd;
        Adam compiledAdam;
        Matrix forward(const Matrix& x, bool training);
        void backward(const Matrix& grad, Optimizer& optimizer, bool canonicalOutputGrad = false);
        void addLayer(int inputs, int outputs, Activation act, Inicialization init, double dropoutRate);
        void train(const Matrix& X, const Matrix& Y, int epochs, Optimizer& optimizer, int batchSize, Loss loss, int shuffleSeed, bool logMetrics, int metricsEvery, int checkpointEvery, const std::string& checkpointDir);
        static void writeOrThrow(std::ofstream& file, const void* data, std::size_t size, const std::string& context);
        static void readOrThrow(std::ifstream& file, void* data, std::size_t size, const std::string& context);
        static void writeMatrixData(std::ofstream& file, const Matrix& mat, const std::string& context);
        static void readMatrixData(std::ifstream& file, Matrix& mat, const std::string& context);
        void saveCheckpoint(const std::string& checkpointDir, int epoch, int totalEpochs, Loss loss, int batchSize, int effectiveBatchSize, int shuffleSeed, bool logMetrics, int metricsEvery, const CheckpointMetrics& metrics);
        static bool supportsCanonicalCrossEntropyGradient(Activation act) { return act == SOFTMAX; }
    public:
        NeuralNetwork();
        NeuralNetwork& input(int features);
        NeuralNetwork& add(const Dense& layer);
        void compile(Loss loss = AUTO_LOSS, double lr = 0.01, int batchSize = 0, int shuffleSeed = -1, bool logMetrics = false, int metricsEvery = 1);
        void compile(Optimizer& optimizer, Loss loss = AUTO_LOSS, int batchSize = 0, int shuffleSeed = -1, bool logMetrics = false, int metricsEvery = 1);
        void fit(const Matrix& X, const Matrix& Y, int epochs);
        void fit(const Matrix& X, const Matrix& Y, int epochs, int checkpointEvery, const std::string& checkpointDir = "checkpoints");
        Matrix predict(const Matrix& x);
        void save(const std::string& path);
        void load(const std::string& path);
};

#endif