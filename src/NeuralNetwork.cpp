#include "NeuralNetwork.h"
#include "Metrics.h"
#include "TrainingUtils.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

void NeuralNetwork::writeOrThrow(std::ofstream& file, const void* data, std::size_t size, const std::string& context) {
    file.write(static_cast<const char*>(data), size);
    if (!file) {
        throw std::runtime_error("NeuralNetwork::save write error: " + context);
    }
}

void NeuralNetwork::readOrThrow(std::ifstream& file, void* data, std::size_t size, const std::string& context) {
    file.read(static_cast<char*>(data), size);
    if (!file) {
        throw std::runtime_error("NeuralNetwork::load read error: " + context);
    }
}

void NeuralNetwork::writeMatrixData(std::ofstream& file, const Matrix& mat, const std::string& context) {
    const std::vector<double>& data = mat.getData();
    if (!data.empty()) {
        writeOrThrow(file, data.data(), data.size() * sizeof(double), context);
    }
}

void NeuralNetwork::readMatrixData(std::ifstream& file, Matrix& mat, const std::string& context) {
    std::vector<double> data((size_t)mat.getRows() * (size_t)mat.getCols());
    if (!data.empty()) {
        readOrThrow(file, data.data(), data.size() * sizeof(double), context);
    }
    mat.setData(data);
}

void NeuralNetwork::saveCheckpoint(const std::string& checkpointDir,
        int epoch,
        int totalEpochs,
        Loss loss,
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
        throw std::runtime_error("NeuralNetwork::train checkpoint directory error: " + dirPath.string());
    }

    std::string modelPath = checkpointDir + ".model";
    std::string ckptPath = checkpointDir + ".ckpt";

    save(modelPath);

    std::ofstream file(ckptPath, std::ios::binary);
    if (!file.is_open()) {
        throw std::invalid_argument("NeuralNetwork::train open checkpoint error: " + ckptPath);
    }

    writeOrThrow(file, kCheckpointMagic, sizeof(kCheckpointMagic), "checkpoint magic");
    int version = kCheckpointVersion;
    writeOrThrow(file, &version, sizeof(int), "checkpoint version");

    writeOrThrow(file, &epoch, sizeof(int), "checkpoint epoch");
    writeOrThrow(file, &totalEpochs, sizeof(int), "checkpoint total epochs");

    int lossVal = static_cast<int>(loss);
    writeOrThrow(file, &lossVal, sizeof(int), "checkpoint loss");
    writeOrThrow(file, &batchSize, sizeof(int), "checkpoint batch size");
    writeOrThrow(file, &effectiveBatchSize, sizeof(int), "checkpoint effective batch size");
    writeOrThrow(file, &shuffleSeed, sizeof(int), "checkpoint shuffle seed");

    int logFlag = logMetrics ? 1 : 0;
    writeOrThrow(file, &logFlag, sizeof(int), "checkpoint log metrics");
    writeOrThrow(file, &metricsEvery, sizeof(int), "checkpoint metrics every");

    writeOrThrow(file, &metrics.type, sizeof(int), "checkpoint metrics type");
    writeOrThrow(file, &metrics.m1, sizeof(double), "checkpoint metric 1");
    writeOrThrow(file, &metrics.m2, sizeof(double), "checkpoint metric 2");
    writeOrThrow(file, &metrics.m3, sizeof(double), "checkpoint metric 3");
    writeOrThrow(file, &metrics.m4, sizeof(double), "checkpoint metric 4");

    file.close();
}

NeuralNetwork::NeuralNetwork():
    inputFeatures(-1),
    hasCompileConfig(false),
    compiledLoss(AUTO_LOSS),
    compiledBatchSize(0),
    compiledShuffleSeed(-1),
    compiledLogMetrics(false),
    compiledMetricsEvery(1),
    compiledOptimizer(nullptr),
    compiledSgd(0.01),
    compiledAdam() {}

NeuralNetwork& NeuralNetwork::input(int features) {
    if (features <= 0) {
        throw std::invalid_argument("NeuralNetwork::input features must be > 0");
    }
    if (!layers.empty()) {
        throw std::logic_error("NeuralNetwork::input must be called before adding layers");
    }

    inputFeatures = features;
    return *this;
}

NeuralNetwork& NeuralNetwork::add(const Dense& layer) {
    if (layer.outputs <= 0) {
        throw std::invalid_argument("NeuralNetwork::add outputs must be > 0");
    }

    if (layers.empty()) {
        if (inputFeatures <= 0) {
            throw std::logic_error("NeuralNetwork::add first layer requires input(features)");
        }
        addLayer(inputFeatures, layer.outputs, layer.act, layer.init, layer.dropoutRate);
    }
    else {
        int inferredInputs = layers.back().getOutputSize();
        addLayer(inferredInputs, layer.outputs, layer.act, layer.init, layer.dropoutRate);
    }

    return *this;
}

void NeuralNetwork::compile(Loss loss, double lr, int batchSize, int shuffleSeed, bool logMetrics, int metricsEvery) {
    if (lr <= 0.0) {
        throw std::invalid_argument("NeuralNetwork::compile learning rate must be > 0");
    }
    if (logMetrics && metricsEvery <= 0) {
        throw std::invalid_argument("NeuralNetwork::compile metricsEvery must be > 0 when logMetrics is enabled");
    }

    compiledLoss = loss;
    compiledBatchSize = batchSize;
    compiledShuffleSeed = shuffleSeed;
    compiledLogMetrics = logMetrics;
    compiledMetricsEvery = metricsEvery;
    compiledSgd.setLearningRate(lr);
    compiledOptimizer = &compiledSgd;
    compiledOptimizer->reset();
    hasCompileConfig = true;
}

void NeuralNetwork::compile(Optimizer& optimizer, Loss loss, int batchSize, int shuffleSeed, bool logMetrics, int metricsEvery) {
    if (logMetrics && metricsEvery <= 0) {
        throw std::invalid_argument("NeuralNetwork::compile metricsEvery must be > 0 when logMetrics is enabled");
    }

    compiledLoss = loss;
    compiledBatchSize = batchSize;
    compiledShuffleSeed = shuffleSeed;
    compiledLogMetrics = logMetrics;
    compiledMetricsEvery = metricsEvery;
    compiledOptimizer = &optimizer;
    compiledOptimizer->reset();
    hasCompileConfig = true;
}

void NeuralNetwork::fit(const Matrix& X, const Matrix& Y, int epochs) {
    if (!hasCompileConfig) {
        throw std::logic_error("NeuralNetwork::fit compile() must be called before fit()");
    }
    if (compiledOptimizer == nullptr) {
        throw std::logic_error("NeuralNetwork::fit invalid optimizer state");
    }

    train(X, Y, epochs, *compiledOptimizer, compiledBatchSize, compiledLoss, compiledShuffleSeed, compiledLogMetrics, compiledMetricsEvery, 0, "");
}

void NeuralNetwork::fit(const Matrix& X, const Matrix& Y, int epochs, int checkpointEvery, const std::string& checkpointDir) {
    if (!hasCompileConfig) {
        throw std::logic_error("NeuralNetwork::fit compile() must be called before fit()");
    }
    if (compiledOptimizer == nullptr) {
        throw std::logic_error("NeuralNetwork::fit invalid optimizer state");
    }

    train(X, Y, epochs, *compiledOptimizer, compiledBatchSize, compiledLoss, compiledShuffleSeed, compiledLogMetrics, compiledMetricsEvery, checkpointEvery, checkpointDir);
}

void NeuralNetwork::addLayer(int inputs, int outputs, Activation act, Inicialization init, double dropoutRate){
    if (inputs <= 0 || outputs <= 0) {
        throw std::invalid_argument("NeuralNetwork::addLayer invalid dimensions: " + std::to_string(inputs) + "x" + std::to_string(outputs));
    }

    if (!layers.empty()) {
        int expectedInputs = layers.back().getOutputSize();
        if (inputs != expectedInputs) {
            throw std::invalid_argument("NeuralNetwork::addLayer incompatible dimensions: expected " + std::to_string(expectedInputs) + " inputs, got " + std::to_string(inputs));
        }
    }

    if (layers.empty()) {
        inputFeatures = inputs;
    }

    layers.push_back(Layer(inputs, outputs, act, init, dropoutRate));
}

Matrix NeuralNetwork::forward(const Matrix& X, bool training){
    Matrix out = X;
    for(Layer& l : layers){
        out = l.forward(out, training);
    }
    return out;
}

void NeuralNetwork::backward(const Matrix& grad, Optimizer& optimizer, bool canonicalOutputGrad){
    if (layers.empty()) {
        return;
    }

    Matrix currentGrad = grad;
    for (int i = (int)layers.size() - 1; i >= 0; i--){
        bool applyActDerivative = true;
        if (canonicalOutputGrad && i == (int)layers.size() - 1) {
            Activation outAct = layers[i].getAct();
            if (supportsCanonicalCrossEntropyGradient(outAct)) {
                applyActDerivative = false;
            }
        }
        currentGrad = layers[i].backward(currentGrad, optimizer, i, applyActDerivative);
    }
}

void NeuralNetwork::train(const Matrix& X, const Matrix& Y, int epochs, Optimizer& optimizer, int batchSize, Loss loss, int shuffleSeed, bool logMetrics, int metricsEvery, int checkpointEvery, const std::string& checkpointDir){
    if (layers.empty()) {
        throw std::invalid_argument("NeuralNetwork::train invalid state: layers=0");
    }

    if (epochs <= 0) {
        throw std::invalid_argument("NeuralNetwork::train epochs must be > 0");
    }
    if (logMetrics && metricsEvery <= 0) {
        throw std::invalid_argument("NeuralNetwork::train metricsEvery must be > 0 when logMetrics is enabled");
    }
    if (checkpointEvery < 0) {
        throw std::invalid_argument("NeuralNetwork::train checkpointEvery must be >= 0");
    }

    int expectedInputs = layers.front().getWeights().getRows();
    int expectedOutputs = layers.back().getOutputSize();

    if (X.getCols() != expectedInputs) {
        throw std::invalid_argument("NeuralNetwork::train input dimension mismatch: " + std::to_string(X.getRows()) + "x" + std::to_string(X.getCols()) + " | expected Nx" + std::to_string(expectedInputs));
    }

    int n = X.getRows();

    if (n <= 0) {
        throw std::invalid_argument("NeuralNetwork::train empty dataset");
    }

    if (Y.getRows() != n || Y.getCols() != expectedOutputs) {
        throw std::invalid_argument("NeuralNetwork::train label dimension mismatch: " + std::to_string(Y.getRows()) + "x" + std::to_string(Y.getCols()) + " | expected " + std::to_string(n) + "x" + std::to_string(expectedOutputs));
    }

    int effectiveBatchSize = TrainingUtils::resolveEffectiveBatchSize(batchSize, n, "NeuralNetwork::train");

    Loss actualLoss = loss;
    if (loss == AUTO_LOSS) {
        Activation lastAct = layers.back().getAct();
        if (lastAct == SOFTMAX) {
            actualLoss = CROSS_ENTROPY;
        }
        else {
            actualLoss = MSE;
        }
    }

    Activation outputActivation = layers.back().getAct();
    if (actualLoss == CROSS_ENTROPY && !supportsCanonicalCrossEntropyGradient(outputActivation)) {
        throw std::invalid_argument("NeuralNetwork::train CROSS_ENTROPY requires or SOFTMAX output activation");
    }

    std::mt19937 gen = TrainingUtils::createShuffleGenerator(shuffleSeed, "NeuralNetwork::train");
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

            Matrix pred = forward(xBatch, true);

            Matrix grad;
            if (actualLoss == CROSS_ENTROPY) {
                grad = (pred - yBatch) * (1.0 / batchN);
            }
            else {
                grad = (pred - yBatch) * (2.0 / (batchN * pred.getCols()));
            }

            bool canonicalOutputGrad = (actualLoss == CROSS_ENTROPY);
            optimizer.step();
            backward(grad, optimizer, canonicalOutputGrad);
        }

        bool shouldLog = logMetrics && (((e + 1) % metricsEvery == 0) || (e == epochs - 1));
        bool shouldCheckpoint = ((checkpointEvery > 0) && ((e + 1) % checkpointEvery == 0)) || (e == epochs - 1);
        bool needMetrics = shouldLog || shouldCheckpoint;

        if (needMetrics) {
            Matrix epochPred = forward(X, false);
            CheckpointMetrics metrics;

            if (actualLoss == CROSS_ENTROPY) {
                double ce = Metrics::crossEntropy(Y, epochPred);
                double acc = Metrics::accuracy(Y, epochPred);
                metrics.type = 2;
                metrics.m1 = ce;
                metrics.m2 = acc;

                if (shouldLog) {
                    std::cout << "[NeuralNetwork] epoch " << (e + 1) << "/" << epochs
                            << " ce=" << ce << " acc=" << acc << "\n";
                }
            }
            else {
                double mse = Metrics::mse(Y, epochPred);
                double mae = Metrics::mae(Y, epochPred);
                double rmse = Metrics::rmse(Y, epochPred);
                double r2 = Metrics::r2Score(Y, epochPred);
                metrics.type = 1;
                metrics.m1 = mse;
                metrics.m2 = mae;
                metrics.m3 = rmse;
                metrics.m4 = r2;

                if (shouldLog) {
                    std::cout << "[NeuralNetwork] epoch " << (e + 1) << "/" << epochs
                            << " mse=" << mse << " mae=" << mae
                            << " rmse=" << rmse << " r2=" << r2 << "\n";
                }
            }

            if (shouldCheckpoint) {
                saveCheckpoint(checkpointDir, e + 1, epochs, actualLoss, batchSize, effectiveBatchSize, shuffleSeed, logMetrics, metricsEvery, metrics);
            }
        }
    }
}

Matrix NeuralNetwork::predict(const Matrix& x){
    if (layers.empty()) {
        throw std::invalid_argument("NeuralNetwork::predict invalid state: layers=0");
    }

    int expectedInputs = layers.front().getWeights().getRows();

    if (x.getCols() != expectedInputs) {
        throw std::invalid_argument("NeuralNetwork::predict input dimension mismatch: " + std::to_string(x.getRows()) + "x" + std::to_string(x.getCols()) + " | expected Nx" + std::to_string(expectedInputs));
    }

    return forward(x, false);
}

void NeuralNetwork::save(const std::string& path){
    std::ofstream file(path, std::ios::binary);
    if(!file.is_open()){
        throw std::invalid_argument("NeuralNetwork::save open file error: " + path);
    }

    writeOrThrow(file, kModelMagic, sizeof(kModelMagic), "magic");
    int version = kModelVersion;
    writeOrThrow(file, &version, sizeof(int), "version");

    int numLayers = static_cast<int>(layers.size());
    writeOrThrow(file, &numLayers, sizeof(int), "num layers");

    for (const Layer& l : layers) {
        int inputs = l.getWeights().getRows();
        int outputs = l.getWeights().getCols();
        int act = static_cast<int>(l.getAct());
        int init = static_cast<int>(l.getInit());
        double dropoutRate = l.getDropoutRate();

        writeOrThrow(file, &inputs, sizeof(int), "layer inputs");
        writeOrThrow(file, &outputs, sizeof(int), "layer outputs");
        writeOrThrow(file, &act, sizeof(int), "layer activation");
        writeOrThrow(file, &init, sizeof(int), "layer init");
        writeOrThrow(file, &dropoutRate, sizeof(double), "layer dropout");

        Matrix weights = l.getWeights();
        writeMatrixData(file, weights, "weights");

        Matrix bias = l.getBias();
        writeMatrixData(file, bias, "bias");
    }

    const SGD* sgdPtr = dynamic_cast<const SGD*>(compiledOptimizer);
    const Adam* adamPtr = dynamic_cast<const Adam*>(compiledOptimizer);
    int hasCompile = (hasCompileConfig && (sgdPtr != nullptr || adamPtr != nullptr)) ? 1 : 0;
    writeOrThrow(file, &hasCompile, sizeof(int), "compile flag");

    if (hasCompile) {
        int lossVal = static_cast<int>(compiledLoss);
        writeOrThrow(file, &lossVal, sizeof(int), "loss");
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
                throw std::runtime_error("NeuralNetwork::save Adam state size mismatch");
            }

            int stateLayers = static_cast<int>(mW.size());
            writeOrThrow(file, &stateLayers, sizeof(int), "adam state layers");

            for (int i = 0; i < stateLayers; i++) {
                int wRows = mW[i].getRows();
                int wCols = mW[i].getCols();
                int vWRows = vW[i].getRows();
                int vWCols = vW[i].getCols();
                if (wRows != vWRows || wCols != vWCols) {
                    throw std::runtime_error("NeuralNetwork::save Adam weight state dimension mismatch");
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
                    throw std::runtime_error("NeuralNetwork::save Adam bias state dimension mismatch");
                }

                writeOrThrow(file, &bRows, sizeof(int), "adam mB rows");
                writeOrThrow(file, &bCols, sizeof(int), "adam mB cols");
                writeMatrixData(file, mB[i], "adam mB data");
                writeMatrixData(file, vB[i], "adam vB data");
            }
        }
    }

    file.close();
}

void NeuralNetwork::load(const std::string& path){
    std::ifstream file(path, std::ios::binary);
    if(!file.is_open()){
        throw std::invalid_argument("NeuralNetwork::load open file error: " + path);
    }

    char magic[4];
    readOrThrow(file, magic, sizeof(magic), "magic or num layers");

    bool hasHeader = std::memcmp(magic, kModelMagic, sizeof(kModelMagic)) == 0;
    int version = 1;
    int numLayers = 0;

    if (hasHeader) {
        readOrThrow(file, &version, sizeof(int), "version");
        if (version < 2) {
            throw std::runtime_error("NeuralNetwork::load unsupported model version");
        }
        readOrThrow(file, &numLayers, sizeof(int), "num layers");
    }
    else {
        std::memcpy(&numLayers, magic, sizeof(int));
    }

    layers.clear();
    inputFeatures = -1;
    hasCompileConfig = false;
    compiledLoss = AUTO_LOSS;
    compiledBatchSize = 0;
    compiledShuffleSeed = -1;
    compiledLogMetrics = false;
    compiledMetricsEvery = 1;
    compiledOptimizer = nullptr;
    compiledSgd.setLearningRate(0.01);
    compiledAdam = Adam();

    for(int i = 0; i<numLayers; i++){
        int inputs = 0;
        int outputs = 0;
        int actVal = 0;
        int initVal = 0;
        double dropoutRate = 0.0;

        readOrThrow(file, &inputs, sizeof(int), "layer inputs");
        readOrThrow(file, &outputs, sizeof(int), "layer outputs");

        if(inputs <= 0 || outputs <= 0){
            throw std::runtime_error("NeuralNetwork::load dimension mismatch: " + std::to_string(inputs) + "x" + std::to_string(outputs) + " | expected rows>0 and cols>0");
        }

        readOrThrow(file, &actVal, sizeof(int), "layer activation");
        readOrThrow(file, &initVal, sizeof(int), "layer init");

        if (hasHeader && version >= 2) {
            readOrThrow(file, &dropoutRate, sizeof(double), "layer dropout");
        }

        addLayer(inputs, outputs, static_cast<Activation>(actVal), static_cast<Inicialization>(initVal), dropoutRate);

        Matrix w(inputs, outputs);
        readMatrixData(file, w, "weights");

        Matrix b(1, outputs);
        readMatrixData(file, b, "bias");

        layers.back().setWeights(w);
        layers.back().setBias(b);
    }

    if (hasHeader && version >= 2) {
        int hasCompile = 0;
        readOrThrow(file, &hasCompile, sizeof(int), "compile flag");
        if (hasCompile) {
            int lossVal = 0;
            readOrThrow(file, &lossVal, sizeof(int), "loss");
            compiledLoss = static_cast<Loss>(lossVal);

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
                    throw std::runtime_error("NeuralNetwork::load Adam state layer count invalid");
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
    }

    file.close();
}