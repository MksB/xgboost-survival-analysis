// generate_churn_data.cpp : Diese Datei enthält die Funktion "main". Hier beginnt und endet die Ausführung des Programms.
//

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <sstream>

struct ChurnData {
    std::vector<int> contract_type;
    std::vector<double> monthly_charges;
    std::vector<int> support_calls;
    std::vector<int> age;
    std::vector<double> random_noise;
    std::vector<int> time;
    std::vector<int> event;
};

ChurnData generate_churn_data(int n_samples = 5000, int seed = 42) {
    std::mt19937 gen(seed);

    // --- Features generieren (wie im Python-Code) ---
    std::discrete_distribution<int> dist_contract({ 0.5, 0.3, 0.2 });
    std::vector<int> contract_type_vec(n_samples);
    for (auto& v : contract_type_vec) v = dist_contract(gen);

    std::normal_distribution<double> dist_monthly(70.0, 30.0);
    std::vector<double> monthly_charges(n_samples);
    for (auto& v : monthly_charges) v = std::clamp(dist_monthly(gen), 20.0, 150.0);

    std::poisson_distribution<int> dist_support(1.5);
    std::vector<int> support_calls(n_samples);
    for (auto& v : support_calls) v = dist_support(gen);

    std::uniform_int_distribution<int> dist_age(18, 79);
    std::vector<int> age(n_samples);
    for (auto& v : age) v = dist_age(gen);

    std::normal_distribution<double> dist_noise(0.0, 1.0);
    std::vector<double> random_noise(n_samples);
    for (auto& v : random_noise) v = dist_noise(gen);

    // --- Hazard-Score & Interaktion ---
    std::vector<double> hazard_score(n_samples, 0.5);
    for (int i = 0; i < n_samples; ++i) {
        hazard_score[i] += contract_type_vec[i] * -1.5;
        hazard_score[i] += monthly_charges[i] * 0.01;
        hazard_score[i] += support_calls[i] * 0.4;
    }
    for (int i = 0; i < n_samples; ++i) {
        if (monthly_charges[i] > 100.0 && support_calls[i] > 3)
            hazard_score[i] += 2.0;
    }

    // --- Risk & wahre Zeit bis Event ---
    const double baseline_hazard = 0.002;
    std::vector<double> risk(n_samples);
    for (int i = 0; i < n_samples; ++i)
        risk[i] = baseline_hazard * std::exp(hazard_score[i]);

    std::vector<double> true_time_to_event(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        std::exponential_distribution<double> exp_dist(risk[i]);
        true_time_to_event[i] = exp_dist(gen);
    }

    // --- Beobachtungsfenster & Zensierung ---
    std::uniform_real_distribution<double> dist_obs(0.0, 730.0);
    std::vector<double> observation_window(n_samples);
    for (auto& v : observation_window) v = dist_obs(gen);

    std::vector<double> observed_time(n_samples);
    std::vector<int> event_occurred(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        if (true_time_to_event[i] < observation_window[i]) {
            observed_time[i] = true_time_to_event[i];
            event_occurred[i] = 1;
        }
        else {
            observed_time[i] = observation_window[i];
            event_occurred[i] = 0;
        }
    }

    // --- Runden & Filter time > 0 ---
    ChurnData result;
    for (int i = 0; i < n_samples; ++i) {
        int t = static_cast<int>(std::round(observed_time[i]));
        if (t > 0) {
            result.contract_type.push_back(contract_type_vec[i]);
            result.monthly_charges.push_back(monthly_charges[i]);
            result.support_calls.push_back(support_calls[i]);
            result.age.push_back(age[i]);
            result.random_noise.push_back(random_noise[i]);
            result.time.push_back(t);
            result.event.push_back(event_occurred[i]);
        }
    }
    return result;
}

// -------------------------- CSV Export --------------------------
bool save_to_csv(const ChurnData& data, const std::string& filename = "churn_data_synthetic.csv") {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Fehler: Datei '" << filename << "' konnte nicht geöffnet werden!\n";
        return false;
    }

    // Header
    file << "contract_type,monthly_charges,support_calls,age,random_noise,time,event\n";

    // Daten zeilenweise schreiben
    const size_t n = data.contract_type.size();
    for (size_t i = 0; i < n; ++i) {
        file << data.contract_type[i] << ','
            << std::fixed << std::setprecision(6) << data.monthly_charges[i] << ','
            << data.support_calls[i] << ','
            << data.age[i] << ','
            << data.random_noise[i] << ','
            << data.time[i] << ','
            << data.event[i] << '\n';
    }

    file.close();
    std::cout << "CSV erfolgreich gespeichert: " << filename
        << " (" << n << " Zeilen)\n";
    return true;
}

// -------------------------- Hauptprogramm --------------------------
int main() {
    ChurnData df_churn = generate_churn_data(5000, 42);

    const size_t num_rows = df_churn.time.size();
    std::cout << "Datensatz Größe: (" << num_rows << ", 7)\n";

    double churn_rate = 0.0;
    for (int e : df_churn.event) churn_rate += e;
    churn_rate /= num_rows;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Zensierungs-Rate: " << (1.0 - churn_rate) * 100 << "%\n\n";

    std::cout << "Erste 5 Zeilen:\n";
    std::cout << "contract_type\tmonthly_charges\tsupport_calls\tage\trandom_noise\ttime\tevent\n";
    for (size_t i = 0; i < std::min<size_t>(5, num_rows); ++i) {
        std::cout << df_churn.contract_type[i] << "\t\t"
            << std::fixed << std::setprecision(4) << df_churn.monthly_charges[i] << "\t\t"
            << df_churn.support_calls[i] << "\t\t"
            << df_churn.age[i] << "\t"
            << df_churn.random_noise[i] << "\t\t"
            << df_churn.time[i] << "\t" << df_churn.event[i] << "\n";
    }
    std::cout << "\n";

    // CSV exportieren
    save_to_csv(df_churn, "churn_data_synthetic.csv");

    return 0;
}///:~