#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <memory>
#include <tuple>

// Simple Decision Tree Node structure
struct Node {
    bool is_leaf = true;
    double value = 0.0;  // Leaf value (mean for regression)
    int feature_index = -1;
    double threshold = 0.0;
    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;
};

// Decision Tree Regressor class (simple CART-like for regression)
class DecisionTreeRegressor {
public:
    DecisionTreeRegressor(int max_depth = 3, int min_samples_split = 2)
        : max_depth_(max_depth), min_samples_split_(min_samples_split) {
    }

    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
        root_ = std::make_unique<Node>();
        build_tree(X, y, *root_, 0);
    }

    std::vector<double> predict(const std::vector<std::vector<double>>& X) const {
        std::vector<double> predictions;
        predictions.reserve(X.size());
        for (const auto& sample : X) {
            predictions.push_back(predict_sample(sample, *root_));
        }
        return predictions;
    }

private:
    std::unique_ptr<Node> root_;
    int max_depth_;
    int min_samples_split_;

    void build_tree(const std::vector<std::vector<double>>& X, const std::vector<double>& y,
        Node& node, int depth) {
        if (depth >= max_depth_ || static_cast<int>(y.size()) < min_samples_split_) {
            node.is_leaf = true;
            node.value = compute_mean(y);
            return;
        }

        double best_gain = -std::numeric_limits<double>::infinity();
        int best_feature = -1;
        double best_threshold = 0.0;
        std::vector<std::vector<double>> best_left_X, best_right_X;
        std::vector<double> best_left_y, best_right_y;

        for (int feature = 0; feature < static_cast<int>(X[0].size()); ++feature) {
            std::vector<std::vector<double>> sorted_X;
            std::vector<double> sorted_y;
            std::tie(sorted_X, sorted_y) = sort_by_feature(X, y, feature);

            for (size_t i = 1; i < sorted_X.size(); ++i) {
                if (sorted_X[i - 1][feature] == sorted_X[i][feature]) continue;

                double threshold = (sorted_X[i - 1][feature] + sorted_X[i][feature]) / 2.0;

                std::vector<std::vector<double>> left_X, right_X;
                std::vector<double> left_y, right_y;
                split_data(sorted_X, sorted_y, feature, threshold, left_X, left_y, right_X, right_y);

                if (left_y.empty() || right_y.empty()) continue;

                double gain = variance_reduction(y, left_y, right_y);
                if (gain > best_gain) {
                    best_gain = gain;
                    best_feature = feature;
                    best_threshold = threshold;
                    best_left_X = std::move(left_X);
                    best_left_y = std::move(left_y);
                    best_right_X = std::move(right_X);
                    best_right_y = std::move(right_y);
                }
            }
        }

        if (best_feature == -1) {
            node.is_leaf = true;
            node.value = compute_mean(y);
            return;
        }

        node.is_leaf = false;
        node.feature_index = best_feature;
        node.threshold = best_threshold;
        node.left = std::make_unique<Node>();
        node.right = std::make_unique<Node>();

        build_tree(best_left_X, best_left_y, *node.left, depth + 1);
        build_tree(best_right_X, best_right_y, *node.right, depth + 1);
    }

    double predict_sample(const std::vector<double>& sample, const Node& node) const {
        if (node.is_leaf) {
            return node.value;
        }
        if (sample[node.feature_index] <= node.threshold) {
            return predict_sample(sample, *node.left);
        }
        else {
            return predict_sample(sample, *node.right);
        }
    }

    static double compute_mean(const std::vector<double>& values) {
        if (values.empty()) return 0.0;
        return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    }

    static double compute_variance(const std::vector<double>& values) {
        if (values.size() < 2) return 0.0;
        double mean = compute_mean(values);
        double sum_sq_diff = 0.0;
        for (double v : values) {
            sum_sq_diff += (v - mean) * (v - mean);
        }
        return sum_sq_diff / values.size();
    }

    static double variance_reduction(const std::vector<double>& parent,
        const std::vector<double>& left,
        const std::vector<double>& right) {
        double parent_var = compute_variance(parent);
        double left_var = compute_variance(left);
        double right_var = compute_variance(right);
        double weighted_child_var = (left.size() * left_var + right.size() * right_var) / parent.size();
        return parent_var - weighted_child_var;
    }

    static std::pair<std::vector<std::vector<double>>, std::vector<double>>
        sort_by_feature(const std::vector<std::vector<double>>& X, const std::vector<double>& y, int feature) {
        std::vector<size_t> indices(X.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
            return X[a][feature] < X[b][feature];
            });

        std::vector<std::vector<double>> sorted_X(X.size());
        std::vector<double> sorted_y(y.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            sorted_X[i] = X[indices[i]];
            sorted_y[i] = y[indices[i]];
        }
        return { sorted_X, sorted_y };
    }

    static void split_data(const std::vector<std::vector<double>>& X, const std::vector<double>& y,
        int feature, double threshold,
        std::vector<std::vector<double>>& left_X, std::vector<double>& left_y,
        std::vector<std::vector<double>>& right_X, std::vector<double>& right_y) {
        left_X.clear(); left_y.clear();
        right_X.clear(); right_y.clear();
        for (size_t i = 0; i < X.size(); ++i) {
            if (X[i][feature] <= threshold) {
                left_X.push_back(X[i]);
                left_y.push_back(y[i]);
            }
            else {
                right_X.push_back(X[i]);
                right_y.push_back(y[i]);
            }
        }
    }
};

// Gradient Boosting Regressor class (using MSE loss by default)
class GradientBoostingRegressor {
public:
    GradientBoostingRegressor(int n_estimators = 100, double learning_rate = 0.1,
        int max_depth = 3, int min_samples_split = 2)
        : n_estimators_(n_estimators), learning_rate_(learning_rate),
        max_depth_(max_depth), min_samples_split_(min_samples_split) {
    }

    void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& y) {
        // Step 1: Initialize F_0 as the mean of y
        initial_prediction_ = std::accumulate(y.begin(), y.end(), 0.0) / y.size();

        // Current predictions start with F_0
        std::vector<double> predictions(y.size(), initial_prediction_);

        for (int t = 0; t < n_estimators_; ++t) {
            // Step 2a: Compute pseudo-residuals (negative gradients for MSE: y - F_{t-1})
            std::vector<double> residuals(y.size());
            for (size_t i = 0; i < y.size(); ++i) {
                residuals[i] = y[i] - predictions[i];
            }

            // Step 2b: Fit a decision tree to the residuals
            DecisionTreeRegressor tree(max_depth_, min_samples_split_);
            tree.fit(X, residuals);

            // Get tree predictions
            std::vector<double> tree_predictions = tree.predict(X);

            // Step 2c: Update the ensemble F_t = F_{t-1} + lr * h_t
            for (size_t i = 0; i < predictions.size(); ++i) {
                predictions[i] += learning_rate_ * tree_predictions[i];
            }

            // Store the tree
            trees_.push_back(std::move(tree));
        }
    }

    std::vector<double> predict(const std::vector<std::vector<double>>& X) const {
        // Step 3: Final prediction starts with F_0
        std::vector<double> predictions(X.size(), initial_prediction_);

        // Add predictions from each tree, scaled by learning rate
        for (const auto& tree : trees_) {
            std::vector<double> tree_predictions = tree.predict(X);
            for (size_t i = 0; i < predictions.size(); ++i) {
                predictions[i] += learning_rate_ * tree_predictions[i];
            }
        }
        return predictions;
    }

private:
    std::vector<DecisionTreeRegressor> trees_;
    int n_estimators_;
    double learning_rate_;
    int max_depth_;
    int min_samples_split_;
    double initial_prediction_ = 0.0;
};

// Example usage
int main() {
    // Sample data: Simple regression example (X with 1 feature, y = 2*X + noise)
    std::vector<std::vector<double>> X = {
        {1.0}, {2.0}, {3.0}, {4.0}, {5.0},
        {6.0}, {7.0}, {8.0}, {9.0}, {10.0}
    };
    std::vector<double> y = { 2.1, 4.2, 5.9, 8.1, 10.0, 11.9, 14.1, 16.0, 17.9, 20.1 };

    GradientBoostingRegressor model(10, 0.1, 2);  // 10 trees, lr=0.1, max_depth=2
    model.fit(X, y);

    // Predict on new data
    std::vector<std::vector<double>> X_test = { {0.5}, {11.0} };
    std::vector<double> predictions = model.predict(X_test);

    std::cout << "Predictions: ";
    for (double p : predictions) {
        std::cout << p << " ";
    }
    std::cout << std::endl;

    return 0;
}///:~

