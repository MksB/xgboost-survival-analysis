// ConsoleApplication1.cpp : Diese Datei enthält die Funktion "main". Hier beginnt und endet die Ausführung des Programms.
//

// Function to find the best split using the Exact Greedy Algorithm from XGBoost (Algorithm 1)
// Includes column subsampling as an optional parameter (subsample_rate between 0 and 1)
// Shrinkage (learning rate eta) is not directly part of split finding but can be applied to leaf weights later

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <random>
#include <numeric>
using namespace std; 

tuple<int, double, double> find_best_split(
    const vector<vector<double>>& data, // n x m feature matrix
    const vector<double>& gradients,    // g_i for each instance
    const vector<double>& hessians,     // h_i for each instance
    const vector<int>& I,               // instance indices in current node
    double lambda,                      // regularization parameter
    double subsample_rate = 1.0         // column subsampling rate (1.0 means no subsampling)
) {
    int n = data.size();    // number of instances (for reference)
    int m = data[0].size(); // number of features

    double G = 0.0, H = 0.0;
    for (int idx : I) {
        G += gradients[idx];
        H += hessians[idx];
    }
    double gamma = (G * G) / (H + lambda); // constant term for gain calculation

    double best_score = 0.0;
    int best_feature = -1;
    double best_threshold = 0.0;

    // Column subsampling: select a random subset of features
    vector<int> features(m);
    iota(features.begin(), features.end(), 0);
    random_device rd;
    mt19937 g(rd());
    shuffle(features.begin(), features.end(), g);
    int num_features = static_cast<int>(m * subsample_rate);

    if (num_features < 1) num_features = 1;

    for (int f = 0; f < num_features; ++f) {
        int k = features[f];

        // Sort instance indices by feature k
        vector<int> sorted_I = I;
        sort(sorted_I.begin(), sorted_I.end(), [&](int a, int b) {
            return data[a][k] < data[b][k];
            });

        double G_L = 0.0, H_L = 0.0;
        for (size_t pos = 0; pos < sorted_I.size() - 1; ++pos) { // splits between instances
            int idx = sorted_I[pos];
            G_L += gradients[idx];
            H_L += hessians[idx];

            double G_R = G - G_L;
            double H_R = H - H_L;

            double score = (G_L * G_L) / (H_L + lambda) + (G_R * G_R) / (H_R + lambda) - gamma;

            if (score > best_score) {
                best_score = score;
                best_feature = k;
                // Threshold: use the value of the current instance for simplicity
                // In practice, handle duplicates and choose midpoint if needed
                best_threshold = data[idx][k];
            }
        }
    }

    return { best_feature, best_threshold, best_score };
}

int main()
{
    vector<vector<double>> data = {
        {1.0,3.0},
        {2.0,2.0},
        {3.0,1.0},
        {4.0,4.0},
    };
    vector<double> gradients = { 1.0,2.0,3.0,4.0 };
    vector<double> hessians = { 1.0,1.0,1.0,1.0 };
    vector<int> I = { 0,1,2,3 };
    double lambda = 1.0;
    double subsample_rate = 0.8;

    auto [feature, threshold, score] = find_best_split(data, gradients, hessians, I, lambda, subsample_rate);

    cout << "Best split: Feature " << feature
        << ", Threshold " << threshold
        << ", Score " << score << endl; 





    return 0;
}///:~

