// Approximate_Algorithm_for_Split_Finding.cpp : Diese Datei enthält die Funktion "main". Hier beginnt und endet die Ausführung des Programms.
//

#include "stdc++.h"
using namespace std; 

struct Instance
{
    vector<double> x; 
    double g, h;
};
vector<double> propose_candidates(const vector<tuple<double, double, double>>& sorted_points, double epsilon)
{
    size_t n = sorted_points.size();
    double total_h = 0.0;
    for (const auto& p : sorted_points) {
        total_h += get<2>(p);
    }
    int approx_num_bins = round(1.0 / epsilon);
    if (approx_num_bins < 2)
    {
        approx_num_bins = 2;
    }
    double step = total_h / approx_num_bins; 

    vector<double> S; 
    if (n == 0)
    {
        return S;
    }
    S.push_back(get<0>(sorted_points[0]));

    double cum = 0.0;
    size_t i = 0;
    for (int b = 1; b < approx_num_bins; ++b)
    {
        double target = b * step; 
        while (i < n && cum < target)
        {
            cum += get<2>(sorted_points[i]);
            ++i;
        }
        if (i > 0 && i < n)
        {
            double s = get<0>(sorted_points[i - 1]);
            if (!S.empty() && S.back() != s)
            {
                S.push_back(s);
            }
        }
    }
    double max_val = get<0>(sorted_points.back());
    if (!S.empty() && S.back() != max_val)
    {
        S.push_back(max_val);
    }
    return S;
}

double calculate_gain(double GL, double HL, double GR, double HR, double total_G,
    double total_H, double lambda)
{
    double parent_score = (total_G * total_G) / (total_H * lambda);
    double child_score = (GL * GL) / (HL + lambda) + (GR * GR) / (HR + lambda);
    return child_score - parent_score;
}

void find_best_approx_split(const vector<Instance>& data, double epsilon, double lambda) {
    size_t n = data.size();
    if (n == 0) {
        cout << "No data" << endl;
        return;
    }
    size_t m = data[0].x.size(); // number of features

    double total_G = 0.0;
    double total_H = 0.0;
    for (const auto& inst : data) {
        total_G += inst.g;
        total_H += inst.h;
    }

    double best_gain = -1e9;
    int best_feat = -1;
    double best_split_val = 0.0;

    for (size_t k = 0; k < m; ++k) {
        vector<tuple<double, double, double>> points(n);
        for (size_t i = 0; i < n; ++i) {
            points[i] = make_tuple(data[i].x[k], data[i].g, data[i].h);
        }
        sort(points.begin(), points.end());

        vector<double> S = propose_candidates(points, epsilon);
        if (S.size() < 2) continue; // not enough for split

        // Build bucket statistics
        vector<double> G_b(S.size() - 1, 0.0);
        vector<double> H_b(S.size() - 1, 0.0);

        size_t j = 0;
        size_t b = 0;
        while (j < n) {
            double curr_x = get<0>(points[j]);
            // Advance bucket if necessary
            while (b < S.size() - 1 && curr_x > S[b + 1]) ++b;
            if (b >= S.size() - 1) break;

            // Sum for same x values
            double sum_g = 0.0;
            double sum_h = 0.0;
            while (j < n && get<0>(points[j]) == curr_x) {
                sum_g += get<1>(points[j]);
                sum_h += get<2>(points[j]);
                ++j;
            }
            G_b[b] += sum_g;
            H_b[b] += sum_h;
        }

        // Find best split for this feature
        double GL = 0.0;
        double HL = 0.0;
        double max_g_for_feat = -1e9;
        double this_split = 0.0;
        for (size_t v = 0; v < G_b.size() - 1; ++v) { // Don't split after last bucket
            GL += G_b[v];
            HL += H_b[v];
            double GR = total_G - GL;
            double HR = total_H - HL;
            if (HL > 0 && HR > 0) { // Avoid empty sides
                double gain = calculate_gain(GL, HL, GR, HR, total_G, total_H, lambda);
                if (gain > max_g_for_feat) {
                    max_g_for_feat = gain;
                    this_split = S[v + 1];
                }
            }
        }

        if (max_g_for_feat > best_gain) {
            best_gain = max_g_for_feat;
            best_feat = k;
            best_split_val = this_split;
        }
    }

    if (best_feat != -1) {
        cout << "Best feature: " << best_feat << ", Split value: " << best_split_val << ", Gain: " << best_gain << endl;
    }
    else {
        cout << "No split found" << endl;
    }
}

int main()
{

    vector<Instance> data = {
        {{1.0,10.0},-1.0,1.0},
        {{2.0,20.0},-1.0,1.0},
        {{3.0,30.0},1.0,1.0},
        {{4.0,40.0},1.0,1.0}
    };

    double epsilon = 0.5;
    double lambda = 1.0;

    find_best_approx_split(data, epsilon, lambda);
    
    return 0;
}///:~

