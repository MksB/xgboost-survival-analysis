// XGBoost_BlackScholes.cpp : Diese Datei enthält die Funktion "main". Hier beginnt und endet die Ausführung des Programms.
//

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <memory>
#include <random>

// Konstanten für Black-Scholes
const double PI = 3.14159265358979323846;


 //Kern-Modul: Black-Scholes Pricing Kernel
 //Dient zur Generierung von Trainingsdaten für die Pipeline.
 
struct BlackScholes {
    static double normalCDF(double x) {
        return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    }

    static double price(double S, double K, double T, double r, double sigma) {
        if (T <= 1e-7) return std::max(S - K, 0.0);
        double d1 = (std::log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
        double d2 = d1 - sigma * std::sqrt(T);
        return S * normalCDF(d1) - K * std::exp(-r * T) * normalCDF(d2);
    }
};


 //Datenstruktur für das Training
 //Optimiert für Cache-Lokalität durch flache Vektoren.

struct Dataset {
    std::vector<std::vector<float>> features; // SoA: Structure of Arrays
    std::vector<float> targets;
    size_t num_samples;
    size_t num_features;

    Dataset(size_t n, size_t f) : num_samples(n), num_features(f) {
        features.resize(f, std::vector<float>(n));
        targets.resize(n);
    }
};


 //Entscheidungsbaum-Knoten
 
struct Node {
    bool is_leaf = false;
    float weight = 0.0f;
    int split_feature = -1;
    float split_threshold = 0.0f;
    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;
};


 //XGBoost Core: Newton-Boosting Regressor

class HighPerfBooster {
private:
    float lambda; // L2 Regularisierung
    float gamma;  // Tree Complexity Penalty
    float eta;    // Learning Rate
    int max_depth;
    std::vector<std::unique_ptr<Node>> trees;

    // Hilfsstruktur für Gradienten-Statistiken
    struct GradStat {
        double sum_g = 0;
        double sum_h = 0;
    };

    
     //Kern-Funktion: Berechnet den optimalen Leaf-Weight nach XGBoost-Formel
     //w = - sum(g) / (sum(h) + lambda)
    
    float compute_weight(const GradStat& stat) const {
        return static_cast<float>(-stat.sum_g / (stat.sum_h + lambda));
    }

    
    //Kern-Funktion: Berechnet den Gain für einen Split
    //Gain = 0.5 * [score_left + score_right - score_total] - gamma
    
    double compute_gain(const GradStat& left, const GradStat& right, const GradStat& total) const {
        auto score = [this](const GradStat& s) -> double {
            return (s.sum_g * s.sum_g) / (s.sum_h + this->lambda);
            };

        double gain = 0.5 * (score(left) + score(right) - score(total)) - gamma;
        return gain;
    }

    std::unique_ptr<Node> build_tree(const Dataset& data, const std::vector<size_t>& indices,
        const std::vector<float>& g, const std::vector<float>& h, int depth) {

        GradStat total_stat;
        for (size_t idx : indices) {
            total_stat.sum_g += g[idx];
            total_stat.sum_h += h[idx];
        }

        std::unique_ptr<Node> node = std::make_unique<Node>();

        // Stop-Kriterien
        if (depth >= max_depth || indices.size() < 2) {
            node->is_leaf = true;
            node->weight = compute_weight(total_stat);
            return node;
        }

        double best_gain = -1e9;
        int best_f = -1;
        float best_threshold = 0;
        std::vector<size_t> best_left_idx, best_right_idx;

        // Exakte Greedy-Suche nach dem besten Split
        for (int f = 0; f < static_cast<int>(data.num_features); ++f) {
            std::vector<size_t> sorted_indices = indices;
            std::sort(sorted_indices.begin(), sorted_indices.end(), [&](size_t a, size_t b) {
                return data.features[f][a] < data.features[f][b];
                });

            GradStat left_stat = { 0.0, 0.0 };
            for (size_t i = 0; i < sorted_indices.size() - 1; ++i) {
                size_t idx = sorted_indices[i];
                left_stat.sum_g += g[idx];
                left_stat.sum_h += h[idx];

                // Überspringe gleiche Werte
                if (data.features[f][idx] == data.features[f][sorted_indices[i + 1]]) continue;

                GradStat right_stat = {
                    total_stat.sum_g - left_stat.sum_g,
                    total_stat.sum_h - left_stat.sum_h
                };

                double gain = compute_gain(left_stat, right_stat, total_stat);

                if (gain > best_gain) {
                    best_gain = gain;
                    best_f = f;
                    best_threshold = (data.features[f][idx] + data.features[f][sorted_indices[i + 1]]) / 2.0f;
                }
            }
        }

        if (best_gain <= 0 || best_f == -1) {
            node->is_leaf = true;
            node->weight = compute_weight(total_stat);
            return node;
        }

        node->split_feature = best_f;
        node->split_threshold = best_threshold;

        std::vector<size_t> left_idx, right_idx;
        for (size_t idx : indices) {
            if (data.features[best_f][idx] <= best_threshold) {
                left_idx.push_back(idx);
            }
            else {
                right_idx.push_back(idx);
            }
        }

        node->left = build_tree(data, left_idx, g, h, depth + 1);
        node->right = build_tree(data, right_idx, g, h, depth + 1);
        return node;
    }

    float predict_single(const Node& node, const std::vector<float>& features) const {
        if (node.is_leaf) return node.weight;
        if (features[node.split_feature] <= node.split_threshold) {
            return predict_single(*node.left, features);
        }
        return predict_single(*node.right, features);
    }

public:
    HighPerfBooster(float l, float g, float e, int d)
        : lambda(l), gamma(g), eta(e), max_depth(d) {
    }

    void train(const Dataset& data, int num_iters) {
        std::vector<float> predictions(data.num_samples, 0.0f);
        std::vector<size_t> all_indices(data.num_samples);
        std::iota(all_indices.begin(), all_indices.end(), 0);

        for (int i = 0; i < num_iters; ++i) {
            std::vector<float> g(data.num_samples), h(data.num_samples);

            // MSE Loss: g = 2*(pred - target), h = 2.0 (für Newton-Boosting)
            for (size_t j = 0; j < data.num_samples; ++j) {
                g[j] = 2.0f * (predictions[j] - data.targets[j]);
                h[j] = 2.0f;
            }

            auto tree = build_tree(data, all_indices, g, h, 0);

            // Update predictions
            for (size_t j = 0; j < data.num_samples; ++j) {
                std::vector<float> sample_features(data.num_features);
                for (size_t f = 0; f < data.num_features; ++f) {
                    sample_features[f] = data.features[f][j];
                }
                predictions[j] += eta * predict_single(*tree, sample_features);
            }

            trees.push_back(std::move(tree));

            // Minimales Logging
            if (i % 10 == 0) {
                std::cout << "Iteration " << i << " abgeschlossen." << std::endl;
            }
        }
    }

    float predict(const std::vector<float>& features) const {
        float res = 0;
        for (const auto& tree : trees) {
            res += eta * predict_single(*tree, features);
        }
        return res;
    }
};


 //Hauptprogramm
 
int main() {
    const size_t train_size = 500;
    const size_t num_features = 4; // S, K, T, sigma (r konstant)

    Dataset train_data(train_size, num_features);

    // Zufallsgenerator initialisieren
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> s_dist(80.0, 120.0);
    std::uniform_real_distribution<> t_dist(0.1, 2.0);
    std::uniform_real_distribution<> sigma_dist(0.1, 0.5);

    std::cout << "Generiere Black-Scholes Trainingsdaten..." << std::endl;
    for (size_t i = 0; i < train_size; ++i) {
        float S = static_cast<float>(s_dist(gen));
        float K = 100.0f;
        float T = static_cast<float>(t_dist(gen));
        float sigma = static_cast<float>(sigma_dist(gen));
        float r = 0.03f;

        train_data.features[0][i] = S;
        train_data.features[1][i] = K;
        train_data.features[2][i] = T;
        train_data.features[3][i] = sigma;
        train_data.targets[i] = static_cast<float>(BlackScholes::price(S, K, T, r, sigma));
    }

    // Booster Konfiguration (Lambda=1.0, Gamma=0.1, Eta=0.1, Depth=5)
    HighPerfBooster booster(1.0f, 0.1f, 0.1f, 5);

    std::cout << "Starte Training (Newton-Boosting)..." << std::endl;
    booster.train(train_data, 50);

    // Validierung
    std::cout << "\nVergleich: Black-Scholes vs. GBDT-Approximation" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    for (int i = 0; i < 5; ++i) {
        float S = 95.0f + i * 2.5f;
        std::vector<float> f = { S, 100.0f, 1.0f, 0.2f };
        double real = BlackScholes::price(S, 100.0, 1.0, 0.03, 0.2);
        double pred = booster.predict(f);

        std::cout << "Spot: " << S << " | Real: " << real
            << " | GBDT: " << pred << " | Diff: " << std::abs(real - pred) << std::endl;
    }

    return 0;
}///:~