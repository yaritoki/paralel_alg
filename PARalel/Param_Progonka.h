#include <omp.h>
#include <iostream>
#include <chrono>
using namespace std;
using namespace std::chrono;

 double q_func(const double x)
{
    return 4 * pow((cos(2 * x)), 2);
}

 double f_func(const double x)
{
    return pow(sin(4 * x), 2) - 8 * cos(4 * x);
}

 double u_func(const double x)
{
    return pow(sin(2 * x), 2);
}
 /*double q_func1(const double x)
 {
     return (1 + x)* (1 + x);
 }

 double f_func1(const double x)
 {
     return 1-6/((1 + x)* (1 + x)* (1 + x)* (1 + x));
 }

 double u_func1(const double x)
 {
     return 1/((1+x)* (1 + x));
 }*/
void ParamProgonka(int n,int p)
{
    

    int   m;
    double  h, A, B, u_a, u_b, a, c;


   

    A = -2;
    B = 2;

   
    h = (B - A) / (n + 2);
    u_a = u_func(A);
    u_b = u_func(B);

    
    m = n / p;

    double* f = new double[n + 1];
    double* u = new double[n + 1];
    double* y = new double[n + 1];
    double* b = new double[n + 1];
    double* x = new double[n + 1];
    for (int i = 0; i <= n; i++)
    {
        x[i] = A + (i + 1) * h;
        u[i] = u_func(x[i]);
        f[i] = h * h * f_func(x[i]);
        b[i] = 2 + h * h * q_func(x[i]);
    }
    a = 1;
   
    c = 1;
    f[0] += u_a;
    f[n] += u_b;

    double max = 0;

    double* alpha = new double[n + 1];
    double* betta = new double[n + 1];


    alpha[0] = c / b[0];
    betta[0] = f[0] / b[0];
    for (int j = 1; j <= n; j++)
    {
        double div = b[j] - a * alpha[j - 1];
        alpha[j] = c / div;
        betta[j] = (f[j] + a * betta[j - 1]) / div;
    }
    y[n] = betta[n];
    for (int j = n - 1; j >= 0; j--)
    {
        y[j] = alpha[j] * y[j + 1] + betta[j];
    }

   
    time_point<system_clock> start = system_clock::now();
    double** v = new double* [p + 1];
    double** z = new double* [p + 1];
    double** w = new double* [p + 1];
    double* x_ = new double[p + 1];

    for (int i = 0; i <= p; i++)
    {
        v[i] = new double[m + 1];
        z[i] = new double[m + 1];
        w[i] = new double[m + 1];
    }
    omp_set_num_threads(p);

#pragma omp parallel
    {
#pragma omp for
        for (int mu = 0; mu <= p - 1; mu++)
        {
            // 1 система
            int id = omp_get_thread_num();
            double* P = new double[m + 1];
            double* Q = new double[m + 1];
            P[1] = c / b[1];
            Q[1] = a / b[1];
            for (int j = 2; j <= m - 1; j++)
            {
                double div = b[j] - a * P[j - 1];
                P[j] = c / div;
                Q[j] = a * Q[j - 1] / div;
            }
            v[mu][0] = 1;
            v[mu][m - 1] = Q[m - 1];
            v[mu][m] = 0;
            for (int j = m - 2; j >= 1; j--)
            {
                v[mu][j] = P[j] * v[mu][j + 1] + Q[j];
            }

            // 2 система
            z[mu][0] = 0;
            if (m - 2 == 0) z[mu][m - 1] = c / b[m-1];
            else z[mu][m - 1] = c / (b[m-1] - a * P[m - 2]);
            z[mu][m] = 1;
            for (int j = m - 2; j >= 1; j--)
            {
                z[mu][j] = P[j] * z[mu][j + 1];
            }

            // 3 система
            Q[1] = f[1 + mu * m] / b[1 + mu * m];
            for (int j = 2; j <= m - 1; j++)
            {
                Q[j] = (f[j + mu * m] + a * Q[j - 1]) / (b[j] - a * P[j - 1]);
            }
            w[mu][0] = 0;
            w[mu][m - 1] = Q[m - 1];
            w[mu][m] = 0;
            for (int j = m - 2; j >= 1; j--)
            {
                w[mu][j] = P[j] * w[mu][j + 1] + Q[j];
            }
        }
    }

    alpha[0] = c * z[0][1] / (b[0] - c * v[0][1]);
    betta[0] = (f[0] + c * w[0][1]) / (b[0] - c * v[0][1]);
#pragma omp parallel
    {
#pragma omp for
        for (int mu = 1; mu <= p; mu++)
        {
            if (mu == p)
            {
                double div = b[mu] - a * z[mu - 1][m - 1] - a * v[mu - 1][m - 1] * alpha[mu - 1];
                betta[mu] = (f[mu * m] + a * w[mu - 1][m - 1] + a * v[mu - 1][m - 1] * betta[mu - 1]) / div;
            }
            else
            {
                double div = b[mu] - a * z[mu - 1][m - 1] - c * v[mu][1] - a * v[mu - 1][m - 1] * alpha[mu - 1];
                alpha[mu] = c * z[mu][1] / div;
                betta[mu] = (f[mu * m] + a * w[mu - 1][m - 1] + c * w[mu][1] + a * v[mu - 1][m - 1] * betta[mu - 1]) / div;
            }
        }
    }
    x_[p] = betta[p];

    for (int mu = p - 1; mu >= 0; mu--)
    {
        x_[mu] = alpha[mu] * x_[mu + 1] + betta[mu];
    }

#pragma omp parallel
    {
#pragma omp for
        for (int mu = 0; mu <= p - 1; mu++)
        {
            for (int j = 0; j <= m - 1; j++)
            {
                x[j + mu * m] = v[mu][j] * x_[mu] + z[mu][j] * x_[mu + 1] + w[mu][j];
            }
        }
    }
    x[n] = x_[p];
    
    //for (int i = 0; i <= n; i++)
    //{
    //    cout << i << " " << x[i] << " " << u[i] << " " << abs(x[i] - u[i]) << endl;
    //    //if (abs(x[i] - u[i]) > max) max = abs(x[i] - u[i]);
    //}
    //for (int i = 0; i <= n; i++)
    //{
    //    cout << i << " " << y[i] << " " << u[i] << " " << abs(y[i] - u[i]) << endl;
    //    //if (abs(x[i] - u[i]) > max) max = abs(x[i] - u[i]);
    //}
    time_point<system_clock> end = system_clock::now();
    for (int i = 0; i <= n; i++)
    {
        if (abs(y[i] - u[i]) > max) max = abs(y[i] - u[i]);
    }
    cout <<"|y_r-y|= "<< max << endl;
   
    auto time = duration_cast<microseconds>(end - start);
    cout <<"param_progonka " << time.count() << " " << clock() << endl;

   

}
