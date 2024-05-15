#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>

//const int T = 8; // threads count

const double a = 0.0;
const double b = 3.0;




double q(const double x)
{
	return 4 * pow((cos(2 * x)), 2);
}

double f(const double x)
{
	return pow(sin(4 * x), 2) - 8 * cos(4 * x);
}

double u(const double x)



{
	return pow(sin(2 * x), 2);
}

std::vector<double> initialize_matrix(int q_reduc, std::vector<double>& left_diag,
	std::vector<double>& mid_diag,
	std::vector<double>& right_diag,
	std::vector<double>& column)
{
	int n = pow(2, q_reduc);
	std::vector<double> grid(n + 1);
	
	
	const double h = (b - a) / n;
#pragma omp parallel for
	for (int i = 1; i < n; i++)
	{
		// #pragma omp ordered
		grid[i] = a + double(i) * h;

		left_diag[i] = 1;
		mid_diag[i] = 2 + h * h * q(grid[i]);
		right_diag[i] = 1;
		column[i] = h * h * f(grid[i]);
	}
	grid[0] = a;
	grid[n] = b;

	left_diag[0] = 0;
	mid_diag[0] = 1;
	right_diag[0] = 0;

	

		left_diag[n] = 0;
	mid_diag[n] = 1;
	right_diag[n] = 0;

	column[0] = u(grid[0]);
	column[n] = u(grid[n]);

	return grid;
}

//std::vector<double> thomas_algorithm(const std::vector<double>& left_diag,
//	const std::vector<double>& mid_diag,
//	const std::vector<double>& right_diag,
//	const std::vector<double>& column)
//{
//	// coefficients
//	std::vector<double> P(n);
//	std::vector<double> Q(n); //
//
//	// forward
//	P[0] = right_diag[0] / mid_diag[0];
//	Q[0] = column[0] / mid_diag[0];
//
//	for (int i = 1; i < n; i++)
//	{
//		P[i] = right_diag[i] / (mid_diag[i] - left_diag[i] * P[i - 1]);
//		Q[i] = (column[i] + left_diag[i] * Q[i - 1]) / (mid_diag[i] - left_diag[i] * P[i - 1]);
//	} //
//
//	
//
//		// back
//		std::vector<double> y(n + 1);
//	// y[0] = u(a);
//	// y[n-1] = u(b);
//	y[n] = (column[n] + left_diag[n] * Q[n - 1]) / (mid_diag[n] - left_diag[n] * P[n - 1]);
//	for (int i = n - 1; i >= 0; i--)
//	{
//		y[i] = P[i] * y[i + 1] + Q[i];
//	} //
//
//	return y;
//}

std::vector<double> cyclic_reduction(int q_reduc, std::vector<double> left_diag,
	std::vector<double> mid_diag,
	std::vector<double> right_diag,
	std::vector<double> column)
{
	int n = pow(2, q_reduc);
	const double h = (b - a) / n;
	// forward
	for (int k = 1; k < q_reduc; k++)
	{
		int k21 = pow(2, k - 1);
		int k1 = pow(2, k);

#pragma omp parallel for
		for (size_t i = k1; i < n; i += k1)
		{

			

				double P = left_diag[i] / mid_diag[i - k21];
			double Q = right_diag[i] / mid_diag[i + k21];

			// critical?
			left_diag[i] = P * left_diag[i - k21];
			mid_diag[i] = mid_diag[i] - P * right_diag[i - k21] - Q * left_diag[i + k21];
			// critical?
			right_diag[i] = Q * right_diag[i + k21];
			// critical?
			column[i] += P * column[i - k21] + Q * column[i + k21];
		}
	}

	// back
	std::vector<double> y(n + 1);
	y[0] = column[0];
	y[n] = column[n];

	for (int k = q_reduc; k > 0; k--)
	{
		int k2 = pow(2, k);
		int k21 = pow(2, k - 1);
#pragma omp parallel for
		for (size_t i = k21; i <= n - k21; i += k2)
		{
			y[i] = (column[i] + left_diag[i] * y[i - k21] + right_diag[i] * y[i + k21])
				/ mid_diag[i];
		}
	}

	

		return y;
}

double norm(const std::vector<double>& grid, const std::vector<double>& y)
{
	double inaccuracy = 0;
#pragma omp parallel for
	for (size_t i = 0; i < y.size(); i++)
	{
		double diff = std::abs(u(grid[i]) - y[i]);
		// std::cout<<diff<<'\n';
		if (diff > inaccuracy)
			inaccuracy = diff;
	}

	return inaccuracy;
}

using namespace std::chrono; // for calcing time
int reduction_start(int q_reduc, int T)
{
	int n = pow(2, q_reduc);
	omp_set_num_threads(T);

	const int launches_count = 100;
	//double thomas_time = 0; // in microseconds
	double cyclic_time = 0; // in microseconds

	

	std::vector<double> left_diag(n + 1);
	std::vector<double> mid_diag(n + 1);
	std::vector<double> right_diag(n + 1);
	std::vector<double> column(n + 1);

	//double inaccuracy_thomas = 0; // mean
	double inaccuracy_reduction = 0; // mean

#pragma omp parallel for
	for (int i = 0; i < launches_count; i++)
	{
	//	// Thomas
		auto grid = initialize_matrix(q_reduc,left_diag, mid_diag, right_diag, column);

	//	auto start = high_resolution_clock::now();
	//	auto y = thomas_algorithm(left_diag, mid_diag, right_diag, column);
	//	auto stop = high_resolution_clock::now();
	//	auto duration = duration_cast<microseconds>(stop - start);
	//	// #pragma omp atomic
	//	thomas_time += duration.count();

	//	inaccuracy_thomas = norm(grid, y); //

		// Cyclic Reduction
		auto	start = high_resolution_clock::now();
		// grid = initialize_matrix(left_diag, mid_diag, right_diag, column);
		auto y = cyclic_reduction(q_reduc,left_diag, mid_diag, right_diag, column);

		

		auto	stop = high_resolution_clock::now();
		auto duration = duration_cast<microseconds>(stop - start);
		// #pragma omp atomic
		cyclic_time += duration.count();

		inaccuracy_reduction = norm(grid, y); //
	}
	//thomas_time /= double(launches_count); // mean
	cyclic_time /= double(launches_count); // mean

	//std::cout << "Number of divisions: " << n << '\n';
	//std::cout << "*****Thomas*****\n";
	//std::cout << " Time: " << thomas_time << "\n";
	//std::cout << " Inaccuracy: " << inaccuracy_thomas << '\n';
	//std::cout << '\n';

	std::cout << "*****Reduction*****\n";
	std::cout << " T: " << T << "\n";
	std::cout << " Time: " << cyclic_time << "\n";
	std::cout << "|y_r-y|= " << inaccuracy_reduction << '\n';

	return 0;
}
