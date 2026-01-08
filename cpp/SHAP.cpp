// SHAP.cpp : Diese Datei enthält die Funktion "main". Hier beginnt und endet die Ausführung des Programms.
//

#include <iostream>
#include "stdc++.h"
#include <nmmintrin.h>
#include <vector>
#include <functional>
#include <cassert>
#include <algorithm>

using namespace std;

const int MAX_M = 20; 
vector<long long> precompute_factorials()
{
    vector<long long> fact(MAX_M + 1, 1);
	for (int i = 1; i <= MAX_M; ++i)
	{
		fact[i] = fact[i - 1] * i;
	}
	return fact;
}
int popcount(unsigned int x) {
	int count = 0;
	while (x) {
		count += x & 1;
		x >>= 1;
	}
	return count;
}

vector<long double> compute_shap_values(
	const vector<double>& x,
	const vector<double>& baseline,
	function<double(const vector<double>&)> model,
	int M
)
{
	assert(M <= MAX_M && "M exceeds maximum allowed for factorial computation");
	assert(x.size() == M && baseline.size() == M);

	static vector<long long> fact = precompute_factorials();
	long long fact_M = fact[M];

	vector<long double> shap(M, 0.0L);

	for (int j = 0; j < M; ++j)
	{
		int N_sub = M - 1;
		for (int mask = 0; mask < (1 << N_sub); ++mask)
		{
			int size_S = popcount(mask);
			long long fact_S = fact[size_S];
			long long fact_rest = fact[M - size_S - 1];
			long double weight = static_cast<long double>(fact_S * fact_rest) / fact_M;

			vector<double> input_S(M, 0.0);
			for (int i = 0; i < M; ++i)
			{
				input_S[i] = baseline[i];
			}
			int bit = 0;
			for (int i = 0; i < M; ++i)
			{
				if (i == j)
				{
					continue;
				}
				if (mask & (i << bit))
				{
					input_S[i] = x[i];
				}
				++bit;
			}
			double f_S = model(input_S);

			vector<double> input_Sj = input_S;
			input_Sj[j] = x[j];
			double f_Sj = model(input_Sj);

			long double delta = static_cast<long double>(f_Sj - f_S);
			shap[j] += weight * delta;
		}
	}
	return shap;
}
double example_model(const vector<double>& v)
{
	double sum = 0.0;
	for (double val : v)
	{
		sum += val;
	}
	return sum;
}

int main()
{
	int M = 3;
	vector<double> x = { 1.0,2.0,3.0 };
	vector<double> baseline = { 0.0,0.0,0.0 };

	auto shap = compute_shap_values(x, baseline, example_model, M);

	cout << "SHAP values: " << endl; 
	for (int j = 0; j < M; ++j)
	{
		cout << "Feature " << j << ": " << shap[j] << endl;
	}


    return 0;
}///:~

