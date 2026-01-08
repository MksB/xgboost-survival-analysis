// Sparsity-aware_split.cpp : Diese Datei enthält die Funktion "main". Hier beginnt und endet die Ausführung des Programms.
//

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

struct SplitResult
{
    int feature_index; 
    float split_value; 
    std::string default_direction;
    float gain; 
};
float ComputeGain(float G_L, float H_L, float G_R, float H_R,
    float G, float H, float lambda)
{
    return (G_L * G_L) / (H_L + lambda) + (G_R * G_R) / (H_R + lambda) - (G * G) / (H + lambda);
}

SplitResult FindBestSplitForFeatur(
    int k,
    const std::vector<std::vector<float>>& X,
    const std::vector<float>& g,
    const std::vector<float>& h,
    float G,
    float H,
    float lambda
)
{
    int n = X.size();
    std::vector<std::pair<float, int>> non_missing;
    for (int i = 0; i < n; ++i)
    {
        if (!std::isnan(X[i][k]))
        {
            non_missing.emplace_back(X[i][k], i);
        }
    }

    float best_score = -std::numeric_limits<float>::infinity();
    float best_split_value = 0.0f;
    std::string best_direction = "";

    std::sort(non_missing.begin(), non_missing.end());

    float G_L = 0.0f;
    float H_L = 0.0f;
    for (const auto& entry : non_missing)
    {
        int j = entry.second;
        G_L += g[j];
        H_L += h[j];

        float G_R = G - G_L;
        float H_R = H - H_L;
        float score = ComputeGain(G_L, H_L, G_R, H_R, G, H, lambda);
        if (score > best_score)
        {
            best_score = score; 
            best_split_value = entry.first;
            best_direction = "right";
        }
    }

    std::sort(non_missing.rbegin(), non_missing.rend());

    float G_R = 0.0f;
    float H_R = 0.0f;
    for (const auto& entry : non_missing)
    {
        int j = entry.second;
        G_R += g[j];
        H_R += h[j];
        float G_L = G - G_R;
        float H_L = H - H_R;
        float score = ComputeGain(G_L, H_L, G_R, H_R, G, H, lambda);
        if (score > best_score)
        {
            best_score = score;
            best_split_value = entry.first;
            best_direction = "left";
        }
    }
    return { k,best_split_value, best_direction, best_score };
}

SplitResult SparsityAwareSplitFinding(
    const std::vector<std::vector<float>>& X,
    const std::vector<float>& g,
    const std::vector<float>& h,
    float lambda = 1.0f
)
{
    int n = X.size();
    int m = X[0].size();

    float G = 0.0f;
    float H = 0.0f;
    for (int i = 0; i < n; ++i)
    {
        G += g[i];
        H += h[i];
    }

    SplitResult best_split = { -1,0.0f,"",-std::numeric_limits<float>::infinity() };

    for (int k = 0; k < m; ++k)    
    {
        SplitResult current = FindBestSplitForFeatur(k, X, g, h, G, H, lambda);
        if (current.gain > best_split.gain)
        {
            best_split = current;
        }
    }
    return best_split;
}

int main()
{
    std::vector<std::vector<float>> X = {
        {1.0f,NAN},
        {2.0f,3.0f},
        {NAN,4.0f},
        {5.0f,6.0f}
    };

    std::vector<float> g = { 0.1f,0.2f,0.3f,0.4f };
    std::vector<float> h = { 1.0f,1.0f,1.0f,1.0f };
    
    float lambda = 1.0f;

    SplitResult result = SparsityAwareSplitFinding(X, g, h, lambda);

    std::cout << "Best feature: " << result.feature_index << std::endl; 
    std::cout << "Best split value: " << result.split_value << std::endl; 
    std::cout << "Default direction for missing: " << result.default_direction << std::endl; 
    std::cout << "Max gain: " << result.gain << std::endl; 

    return 0; 
}///:~

