#include "LinearRegression.h"
#include "Metrics.h"
#include "Random.h"
#include "TrainingUtils.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

void LinearRegression::writeOrThrow(std::ofstream& file, const void* data, std::size_t size, const std::string& context) {
    file.write(static_cast<const char*>(data), size);
    if (!file) {
        throw std::runtime_error("LinearRegression::save write error: " + context);
    }
}

void LinearRegression::readOrThrow(std::ifstream& file, void* data, std::size_t size, const std::string& context) {
    file.read(static_cast<char*>(data), size);
    if (!file) {
        throw std::runtime_error("LinearRegression::load read error: " + context);
    }
}

void LinearRegression::writeMatrixData(std::ofstream& file, const Matrix& mat, const std::string& context) {
    const std::vector<double>& data = mat.getData();
    if (!data.empty()) {
        writeOrThrow(file, data.data(), data.size() * sizeof(double), context);
    }
}

void LinearRegression::readMatrixData(std::ifstream& file, Matrix& mat, const std::string& context) {
    std::vector<double> data((size_t)mat.getRows() * (size_t)mat.getCols());
    if (!data.empty()) {
        readOrThrow(file, data.data(), data.size() * sizeof(double), context);
    }
    mat.setData(data);
}

void LinearRegression::saveCheckpoint(const std::string& checkpointDir,
        int epoch,
        int totalEpochs,
        int batchSize,
        int effectiveBatchSize,
        int shuffleSeed,
        bool logMetrics,
        int metricsEvery,
        const CheckpointMetrics& metrics) {
    std::filesystem::path dirPath = checkpointDir.empty() ? std::filesystem::path(".") : std::filesystem::path(checkpointDir);
    std::error_code ec;
    std::filesystem::create_directories(dirPath, ec);
    if (ec) {
        throw std::runtime_error("LinearRegression::train checkpoint directory error: " + dirPath.string());
    }

    std::string baseName = "epoch_" + std::to_string(epoch);
    std::filesystem::path basePath = dirPath / baseName;
    std::string modelPath = basePath.string() + ".model";
    std::string ckptPath = basePath.string() + ".ckpt";

    save(modelPath);

    std::ofstream file(ckptPath, std::ios::binary);
    if (!file.is_open()) {
        throw std::invalid_argument("LinearRegression::train open checkpoint error: " + ckptPath);
    }

    writeOrThrow(file, kCheckpointMagic, sizeof(kCheckpointMagic), "checkpoint magic");
    int version = kCheckpointVersion;
    writeOrThrow(file, &version, sizeof(int), "checkpoint version");

    writeOrThrow(file, &epoch, sizeof(int), "checkpoint epoch");
    writeOrThrow(file, &totalEpochs, sizeof(int), "checkpoint total epochs");

    writeOrThrow(file, &batchSize, sizeof(int), "checkpoint batch size");
    writeOrThrow(file, &effectiveBatchSize, sizeof(int), "checkpoint effective batch size");
    writeOrThrow(file, &shuffleSeed, sizeof(int), "checkpoint shuffle seed");

    int logFlag = logMetrics ? 1 : 0;
    writeOrThrow(file, &logFlag, sizeof(int), "checkpoint log metrics");
    writeOrThrow(file, &metricsEvery, sizeof(int), "checkpoint metrics every");

    writeOrThrow(file, &metrics.mse, sizeof(double), "checkpoint mse");
    writeOrThrow(file, &metrics.mae, sizeof(double), "checkpoint mae");
    writeOrThrow(file, &metrics.rmse, sizeof(double), "checkpoint rmse");
    writeOrThrow(file, &metrics.r2, sizeof(double), "checkpoint r2");

    file.close();
}

LinearRegression::LinearRegression(int inputs):
    weights(inputs, 1),
    bias(0.0),
    hasCompileConfig(false),
    compiledBatchSize(0),
    compiledShuffleSeed(-1),
    compiledLogMetrics(false),
    compiledMetricsEvery(1),
    compiledOptimizer(nullptr),
    compiledSgd(0.01),
    compiledAdam() {
    if (inputs <= 0) {
        throw std::invalid_argument("LinearRegression::LinearRegression invalid input dimension: " + std::to_string(inputs));
    }

    Random rng;
    for (int i = 0; i < inputs; i++) {
        weights(i, 0) = rng.uniform(-0.01, 0.01);
    }
}

void LinearRegression::compile(double lr, int batchSize, int shuffleSeed, bool logMetrics, int metricsEvery) {
    if (lr <= 0.0) {
        throw std::invalid_argument("LinearRegression::compile learning rate must be > 0");
    }
    if (logMetrics && metricsEvery <= 0) {
        throw std::invalid_argument("LinearRegression::compile metricsEvery must be > 0 when logMetrics is enabled");
    }

    compiledBatchSize = batchSize;
    compiledShuffleSeed = shuffleSeed;
    compiledLogMetrics = logMetrics;
    compiledMetricsEvery = metricsEvery;
    compiledSgd.setLearningRate(lr);
    compiledOptimizer = &compiledSgd;
    compiledOptimizer->reset();
    hasCompileConfig = true;
}

void LinearRegression::compile(Optimizer& optimizer, int batchSize, int shuffleSeed, bool logMetrics, int metricsEvery) {
    if (logMetrics && metricsEvery <= 0) {
        throw std::invalid_argument("LinearRegression::compile metricsEvery must be > 0 when logMetrics is enabled");
    }

    compiledBatchSize = batchSize;
    compiledShuffleSeed = shuffleSeed;
    compiledLogMetrics = logMetrics;
    compiledMetricsEvery = metricsEvery;
    compiledOptimizer = &optimizer;
    compiledOptimizer->reset();
    hasCompileConfig = true;
}

void LinearRegression::fit(const Matrix& X, const Matrix& Y, int epochs) {
    if (!hasCompileConfig) {
        throw std::logic_error("LinearRegression::fit compile() must be called before fit()");
    }
    if (compiledOptimizer == nullptr) {
        throw std::logic_error("LinearRegression::fit invalid optimizer state");
    }

    train(X, Y, epochs, *compiledOptimizer, compiledBatchSize, compiledShuffleSeed, compiledLogMetrics, compiledMetricsEvery, 0, "");
}

void LinearRegression::fit(const Matrix& X, const Matrix& Y, int epochs, int checkpointEvery, const std::string& checkpointDir) {
    if (!hasCompileConfig) {
        throw std::logic_error("LinearRegression::fit compile() must be called before fit()");
    }
    if (compiledOptimizer == nullptr) {
        throw std::logic_error("LinearRegression::fit invalid optimizer state");
    }

    train(X, Y, epochs, *compiledOptimizer, compiledBatchSize, compiledShuffleSeed, compiledLogMetrics, compiledMetricsEvery, checkpointEvery, checkpointDir);
}

Matrix LinearRegression::forwardLinear(const Matrix& X) const {
    Matrix pred = X.dot(weights);
    double b = bias;
    for (int i = 0; i < pred.getRows(); i++) {
        pred(i, 0) += b;
    }
    return pred;
}

double LinearRegression::sumColumnVector(const Matrix& vec) const {
    if (vec.getCols() != 1) {
        throw std::invalid_argument("LinearRegression::sumColumnVector expected a column vector");
    }

    double sum = 0.0;
    for (int i = 0; i < vec.getRows(); i++) {
        sum += vec(i, 0);
    }
    return sum;
}

void LinearRegression::train(const Matrix& X, const Matrix& Y, int epochs, Optimizer& optimizer, int batchSize, int shuffleSeed, bool logMetrics, int metricsEvery, int checkpointEvery, const std::string& checkpointDir){
    int expectedInputs = weights.getRows();

    if (epochs <= 0) {
        throw std::invalid_argument("LinearRegression::train epochs must be > 0");
    }
    if (logMetrics && metricsEvery <= 0) {
        throw std::invalid_argument("LinearRegression::train metricsEvery must be > 0 when logMetrics is enabled");
    }
    if (checkpointEvery < 0) {
        throw std::invalid_argument("LinearRegression::train checkpointEvery must be >= 0");
    }

    if (X.getCols() != expectedInputs) {
        throw std::invalid_argument("LinearRegression::train input dimension mismatch: " + std::to_string(X.getRows()) + "x" + std::to_string(X.getCols()) + " | expected Nx" + std::to_string(expectedInputs));
    }

    int n = X.getRows();

    if (Y.getRows() != n || Y.getCols() != 1) {
        throw std::invalid_argument("LinearRegression::train label dimension mismatch: " + std::to_string(Y.getRows()) + "x" + std::to_string(Y.getCols()) + " | expected " + std::to_string(n) + "x1");
    }

    if (n <= 0) {
        throw std::invalid_argument("LinearRegression::train empty dataset");
    }

    int effectiveBatchSize = TrainingUtils::resolveEffectiveBatchSize(batchSize, n, "LinearRegression::train");
    std::mt19937 gen = TrainingUtils::createShuffleGenerator(shuffleSeed, "LinearRegression::train");
    std::vector<int> rowOrder = TrainingUtils::makeSequentialRowOrder(n);

    for (int e = 0; e < epochs; e++) {
        TrainingUtils::shuffleRowOrder(rowOrder, gen);

        Matrix xShuffled = X.shuffleRows(rowOrder);
        Matrix yShuffled = Y.shuffleRows(rowOrder);

        for (int start = 0; start < n; start += effectiveBatchSize) {
            int end = start + effectiveBatchSize;
            if (end > n) {
                end = n;
            }

            Matrix xBatch = xShuffled.vslice(start, end);
            Matrix yBatch = yShuffled.vslice(start, end);
            int batchN = xBatch.getRows();

            Matrix pred = forwardLinear(xBatch);
            Matrix grad = (pred - yBatch) * (2.0 / batchN);

            Matrix gradW = xBatch.transpose().dot(grad);
            double gradB = sumColumnVector(grad);

            optimizer.step();

            Matrix biasMat(1, 1);
            biasMat(0, 0) = bias;
            Matrix gradBMat(1, 1);
            gradBMat(0, 0) = gradB;

            optimizer.update(0, weights, biasMat, gradW, gradBMat);
            bias = biasMat(0, 0);
        }

        bool shouldLog = logMetrics && (((e + 1) % metricsEvery == 0) || (e == epochs - 1));
        bool shouldCheckpoint = ((checkpointEvery > 0) && ((e + 1) % checkpointEvery == 0)) || (e == epochs - 1);
        bool needMetrics = shouldLog || shouldCheckpoint;

        if (needMetrics) {
            Matrix epochPred = forwardLinear(X);
            double mse = Metrics::mse(Y, epochPred);
            double mae = Metrics::mae(Y, epochPred);
            double rmse = Metrics::rmse(Y, epochPred);
            double r2 = Metrics::r2Score(Y, epochPred);

            if (shouldLog) {
                std::cout << "[LinearRegression] epoch " << (e + 1) << "/" << epochs
                            << " mse=" << mse << " mae=" << mae
                            << " rmse=" << rmse << " r2=" << r2 << "\n";
            }

            if (shouldCheckpoint) {
                CheckpointMetrics metrics;
                metrics.mse = mse;
                metrics.mae = mae;
                metrics.rmse = rmse;
                metrics.r2 = r2;
                saveCheckpoint(checkpointDir, e + 1, epochs, batchSize, effectiveBatchSize, shuffleSeed, logMetrics, metricsEvery, metrics);
            }
        }
    }

}

Matrix LinearRegression::predict(const Matrix& x){
    int expectedInputs = weights.getRows();

    if (x.getCols() != expectedInputs) {
        throw std::invalid_argument("LinearRegression::predict input dimension mismatch: " + std::to_string(x.getRows()) + "x" + std::to_string(x.getCols()) + " | expected Nx" + std::to_string(expectedInputs));
    }

    return forwardLinear(x);
}

void LinearRegression::save(const std::string& path){
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::invalid_argument("LinearRegression::save open file error: " + path);
    }

    writeOrThrow(file, kModelMagic, sizeof(kModelMagic), "magic");
    int version = kModelVersion;
    writeOrThrow(file, &version, sizeof(int), "version");

    int rows = weights.getRows();
    int cols = weights.getCols();
    writeOrThrow(file, &rows, sizeof(int), "weights rows");
    writeOrThrow(file, &cols, sizeof(int), "weights cols");
    writeMatrixData(file, weights, "weights");
    writeOrThrow(file, &bias, sizeof(double), "bias");

    const SGD* sgdPtr = dynamic_cast<const SGD*>(compiledOptimizer);
    const Adam* adamPtr = dynamic_cast<const Adam*>(compiledOptimizer);
    int hasCompile = (hasCompileConfig && (sgdPtr != nullptr || adamPtr != nullptr)) ? 1 : 0;
    writeOrThrow(file, &hasCompile, sizeof(int), "compile flag");

    if (hasCompile) {
        writeOrThrow(file, &compiledBatchSize, sizeof(int), "batch size");
        writeOrThrow(file, &compiledShuffleSeed, sizeof(int), "shuffle seed");

        int logFlag = compiledLogMetrics ? 1 : 0;
        writeOrThrow(file, &logFlag, sizeof(int), "log metrics");
        writeOrThrow(file, &compiledMetricsEvery, sizeof(int), "metrics every");

        int optimizerTag = 0;
        if (adamPtr) {
            optimizerTag = 2;
        }
        else if (sgdPtr) {
            optimizerTag = 1;
        }
        writeOrThrow(file, &optimizerTag, sizeof(int), "optimizer tag");

        if (optimizerTag == 1) {
            double lr = sgdPtr->getLearningRate();
            writeOrThrow(file, &lr, sizeof(double), "sgd lr");
        }
        else if (optimizerTag == 2) {
            double lr = adamPtr->getLearningRate();
            double beta1 = adamPtr->getBeta1();
            double beta2 = adamPtr->getBeta2();
            double eps = adamPtr->getEps();
            int t = adamPtr->getT();

            writeOrThrow(file, &lr, sizeof(double), "adam lr");
            writeOrThrow(file, &beta1, sizeof(double), "adam beta1");
            writeOrThrow(file, &beta2, sizeof(double), "adam beta2");
            writeOrThrow(file, &eps, sizeof(double), "adam eps");
            writeOrThrow(file, &t, sizeof(int), "adam t");

            const std::vector<Matrix>& mW = adamPtr->getMW();
            const std::vector<Matrix>& vW = adamPtr->getVW();
            const std::vector<Matrix>& mB = adamPtr->getMB();
            const std::vector<Matrix>& vB = adamPtr->getVB();

            if (mW.size() != vW.size() || mW.size() != mB.size() || mW.size() != vB.size()) {
                throw std::runtime_error("LinearRegression::save Adam state size mismatch");
            }

            int stateLayers = static_cast<int>(mW.size());
            writeOrThrow(file, &stateLayers, sizeof(int), "adam state layers");

            for (int i = 0; i < stateLayers; i++) {
                int wRows = mW[i].getRows();
                int wCols = mW[i].getCols();
                int vWRows = vW[i].getRows();
                int vWCols = vW[i].getCols();
                if (wRows != vWRows || wCols != vWCols) {
                    throw std::runtime_error("LinearRegression::save Adam weight state dimension mismatch");
                }

                writeOrThrow(file, &wRows, sizeof(int), "adam mW rows");
                writeOrThrow(file, &wCols, sizeof(int), "adam mW cols");
                writeMatrixData(file, mW[i], "adam mW data");
                writeMatrixData(file, vW[i], "adam vW data");

                int bRows = mB[i].getRows();
                int bCols = mB[i].getCols();
                int vBRows = vB[i].getRows();
                int vBCols = vB[i].getCols();
                if (bRows != vBRows || bCols != vBCols) {
                    throw std::runtime_error("LinearRegression::save Adam bias state dimension mismatch");
                }

                writeOrThrow(file, &bRows, sizeof(int), "adam mB rows");
                writeOrThrow(file, &bCols, sizeof(int), "adam mB cols");
                writeMatrixData(file, mB[i], "adam mB data");
                writeMatrixData(file, vB[i], "adam vB data");
            }
        }
    }
}

void LinearRegression::load(const std::string& path){
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("LinearRegression::load failed to open file: " + path);
    }

    char magic[4];
    readOrThrow(file, magic, sizeof(magic), "magic or inputs");

    bool hasHeader = std::memcmp(magic, kModelMagic, sizeof(kModelMagic)) == 0;
    int version = 1;

    hasCompileConfig = false;
    compiledBatchSize = 0;
    compiledShuffleSeed = -1;
    compiledLogMetrics = false;
    compiledMetricsEvery = 1;
    compiledOptimizer = nullptr;
    compiledSgd.setLearningRate(0.01);
    compiledAdam = Adam();

    if (hasHeader) {
        readOrThrow(file, &version, sizeof(int), "version");
        if (version < 2) {
            throw std::runtime_error("LinearRegression::load unsupported model version");
        }

        int rows = 0;
        int cols = 0;
        readOrThrow(file, &rows, sizeof(int), "weights rows");
        readOrThrow(file, &cols, sizeof(int), "weights cols");
        if (rows <= 0 || cols <= 0) {
            throw std::runtime_error("LinearRegression::load invalid weight dimensions");
        }

        weights = Matrix(rows, cols);
        readMatrixData(file, weights, "weights");
        readOrThrow(file, &bias, sizeof(double), "bias");

        int hasCompile = 0;
        readOrThrow(file, &hasCompile, sizeof(int), "compile flag");
        if (hasCompile) {
            readOrThrow(file, &compiledBatchSize, sizeof(int), "batch size");
            readOrThrow(file, &compiledShuffleSeed, sizeof(int), "shuffle seed");

            int logFlag = 0;
            readOrThrow(file, &logFlag, sizeof(int), "log metrics");
            compiledLogMetrics = logFlag != 0;
            readOrThrow(file, &compiledMetricsEvery, sizeof(int), "metrics every");

            int optimizerTag = 0;
            readOrThrow(file, &optimizerTag, sizeof(int), "optimizer tag");

            if (optimizerTag == 1) {
                double lr = 0.0;
                readOrThrow(file, &lr, sizeof(double), "sgd lr");
                compiledSgd.setLearningRate(lr);
                compiledOptimizer = &compiledSgd;
                hasCompileConfig = true;
            }
            else if (optimizerTag == 2) {
                double lr = 0.0;
                double beta1 = 0.0;
                double beta2 = 0.0;
                double eps = 0.0;
                int t = 0;

                readOrThrow(file, &lr, sizeof(double), "adam lr");
                readOrThrow(file, &beta1, sizeof(double), "adam beta1");
                readOrThrow(file, &beta2, sizeof(double), "adam beta2");
                readOrThrow(file, &eps, sizeof(double), "adam eps");
                readOrThrow(file, &t, sizeof(int), "adam t");

                int stateLayers = 0;
                readOrThrow(file, &stateLayers, sizeof(int), "adam state layers");
                if (stateLayers < 0) {
                    throw std::runtime_error("LinearRegression::load Adam state layer count invalid");
                }

                std::vector<Matrix> mW;
                std::vector<Matrix> vW;
                std::vector<Matrix> mB;
                std::vector<Matrix> vB;
                mW.reserve(stateLayers);
                vW.reserve(stateLayers);
                mB.reserve(stateLayers);
                vB.reserve(stateLayers);

                for (int i = 0; i < stateLayers; i++) {
                    int wRows = 0;
                    int wCols = 0;
                    int bRows = 0;
                    int bCols = 0;

                    readOrThrow(file, &wRows, sizeof(int), "adam mW rows");
                    readOrThrow(file, &wCols, sizeof(int), "adam mW cols");
                    Matrix mWMat(wRows, wCols);
                    readMatrixData(file, mWMat, "adam mW data");

                    Matrix vWMat(wRows, wCols);
                    readMatrixData(file, vWMat, "adam vW data");

                    readOrThrow(file, &bRows, sizeof(int), "adam mB rows");
                    readOrThrow(file, &bCols, sizeof(int), "adam mB cols");
                    Matrix mBMat(bRows, bCols);
                    readMatrixData(file, mBMat, "adam mB data");

                    Matrix vBMat(bRows, bCols);
                    readMatrixData(file, vBMat, "adam vB data");

                    mW.push_back(mWMat);
                    vW.push_back(vWMat);
                    mB.push_back(mBMat);
                    vB.push_back(vBMat);
                }

                compiledAdam = Adam(lr, beta1, beta2, eps);
                compiledAdam.loadState(t, mW, vW, mB, vB);
                compiledOptimizer = &compiledAdam;
                hasCompileConfig = true;
            }
        }

        return;
    }

    int inputs = 0;
    std::memcpy(&inputs, magic, sizeof(int));
    if (inputs <= 0) {
        throw std::runtime_error("LinearRegression::load invalid input dimension");
    }

    weights = Matrix(inputs, 1);
    std::vector<double> wData((size_t)inputs);
    readOrThrow(file, wData.data(), wData.size() * sizeof(double), "legacy weights");
    weights.setData(wData);

    readOrThrow(file, &bias, sizeof(double), "legacy bias");
}