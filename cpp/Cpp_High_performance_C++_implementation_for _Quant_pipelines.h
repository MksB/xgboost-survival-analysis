#include <xgboost/c_api.h>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

// Helper function to check XGBoost API calls for errors
void CheckXGBoostStatus(int status, const std::string& operation) {
    if (status != 0) {
        throw std::runtime_error(operation + " failed: " + XGGetLastError());
    }
}

int main() {
    try {
        // Sample data for quantitative finance example: Predicting stock returns
        // Features: [lag1_return, volume_change, volatility]
        // Labels: next_day_return
        // Toy dataset with 10 training samples, 3 features
        const int train_rows = 10;
        const int num_features = 3;
        std::vector<float> train_data = {
            0.01f, 0.05f, 0.02f,   // Sample 1
            0.02f, 0.03f, 0.015f,  // Sample 2
            -0.01f, -0.02f, 0.025f,// Sample 3
            0.03f, 0.04f, 0.01f,   // Sample 4
            0.005f, 0.01f, 0.03f,  // Sample 5
            -0.005f, -0.01f, 0.035f,// Sample 6
            0.015f, 0.06f, 0.018f, // Sample 7
            0.025f, 0.035f, 0.012f,// Sample 8
            -0.015f, -0.025f, 0.028f,// Sample 9
            0.035f, 0.045f, 0.008f // Sample 10
        };
        std::vector<float> train_labels = {
            0.02f, 0.015f, -0.005f, 0.025f, 0.01f,
            -0.01f, 0.018f, 0.012f, -0.012f, 0.028f
        };

        // Create DMatrix for training data
        DMatrixHandle dtrain;
        int status = XGDMatrixCreateFromMat(train_data.data(), train_rows, num_features, -1.0f, &dtrain);
        CheckXGBoostStatus(status, "XGDMatrixCreateFromMat (train)");
        status = XGDMatrixSetFloatInfo(dtrain, "label", train_labels.data(), train_labels.size());
        CheckXGBoostStatus(status, "XGDMatrixSetFloatInfo (labels)");

        // Create booster
        BoosterHandle booster;
        const DMatrixHandle dmatrices[] = {dtrain};
        status = XGBoosterCreate(dmatrices, 1, &booster);
        CheckXGBoostStatus(status, "XGBoosterCreate");

        // Set parameters for high performance in quant pipelines:
        // - Use histogram-based tree method for faster training
        // - Multi-threading with OpenMP (set nthread to available cores)
        // - Regularization to prevent overfitting in financial data
        // - Objective: regression for return prediction
        status = XGBoosterSetParam(booster, "objective", "reg:squarederror");
        CheckXGBoostStatus(status, "SetParam objective");
        status = XGBoosterSetParam(booster, "tree_method", "hist");  // Fast histogram method
        CheckXGBoostStatus(status, "SetParam tree_method");
        status = XGBoosterSetParam(booster, "nthread", "8");  // Adjust to your CPU cores for parallelism
        CheckXGBoostStatus(status, "SetParam nthread");
        status = XGBoosterSetParam(booster, "eta", "0.1");    // Learning rate
        CheckXGBoostStatus(status, "SetParam eta");
        status = XGBoosterSetParam(booster, "max_depth", "3"); // Shallow trees for speed
        CheckXGBoostStatus(status, "SetParam max_depth");
        status = XGBoosterSetParam(booster, "subsample", "0.8"); // Subsampling for faster training
        CheckXGBoostStatus(status, "SetParam subsample");
        status = XGBoosterSetParam(booster, "lambda", "1.0"); // L2 regularization
        CheckXGBoostStatus(status, "SetParam lambda");

        // Train the model (100 rounds; in quant pipelines, monitor early stopping with validation set)
        std::cout << "Training XGBoost model..." << std::endl;
        for (int iter = 0; iter < 100; ++iter) {
            status = XGBoosterUpdateOneIter(booster, iter, dtrain);
            CheckXGBoostStatus(status, "XGBoosterUpdateOneIter");
        }
        std::cout << "Training complete." << std::endl;

        // Sample test data (2 samples)
        const int test_rows = 2;
        std::vector<float> test_data = {
            0.02f, 0.04f, 0.015f,  // Test sample 1
            -0.01f, -0.03f, 0.025f // Test sample 2
        };

        // Create DMatrix for test data
        DMatrixHandle dtest;
        status = XGDMatrixCreateFromMat(test_data.data(), test_rows, num_features, -1.0f, &dtest);
        CheckXGBoostStatus(status, "XGDMatrixCreateFromMat (test)");

        // Predict
        bst_ulong out_len;
        const float* out_result;
        status = XGBoosterPredict(booster, dtest, 0, 0, 0, &out_len, &out_result);
        CheckXGBoostStatus(status, "XGBoosterPredict");

        // Output predictions
        std::cout << "Predictions for test samples:" << std::endl;
        for (bst_ulong i = 0; i < out_len; ++i) {
            std::cout << "Sample " << (i + 1) << ": Predicted return = " << out_result[i] << std::endl;
        }

        // Cleanup
        XGDMatrixFree(dtrain);
        XGDMatrixFree(dtest);
        XGBoosterFree(booster);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}