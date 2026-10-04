#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

struct Point { double x; double y; };

int main(int argc, char** argv) 
{
    if (argc < 5) 
    {
        cerr << "Usage: " << argv[0] << " <input_file> <num_clusters> <num_threads> <schedule>\n";
        cerr << "       schedule: 'static' or 'dynamic'\n";
        return 1;
    }
    string input_file = argv[1];
    int K;
    int num_threads;
    string schedule_type = argv[4];
    for (char &c : schedule_type) 
    {
        c = tolower(c);
    }
    try 
    {
        K = stoi(argv[2]);
        num_threads = stoi(argv[3]);
    } 
    catch (exception &e) 
    {
        cerr << "Error: num_clusters and num_threads must be integers\n";
        return 1;
    }
    if (K <= 0) 
    {
        cerr << "Error: num_clusters must be positive\n";
        return 1;
    }
    if (num_threads <= 0) 
    {
        cerr << "Error: num_threads must be positive\n";
        return 1;
    }
    if (schedule_type != "static" && schedule_type != "dynamic") 
    {
        cerr << "Error: schedule must be 'static' or 'dynamic'\n";
        return 1;
    }

    // Reading data points from file
    ifstream fin(input_file);
    if (!fin.is_open()) 
    {
        cerr << "Error: Unable to open file " << input_file << "\n";
        return 1;
    }
    vector<Point> points;
    points.reserve(100000);
    string line;
    while (getline(fin, line)) 
    {
        if (line.empty()) continue;
        size_t commaPos = line.find(',');
        if (commaPos == string::npos) continue;
        string sx = line.substr(0, commaPos);
        string sy = line.substr(commaPos + 1);
        try 
        {
            double x = stod(sx);
            double y = stod(sy);
            points.push_back({x, y});
        } 
        catch (exception &e) 
        {
            continue;
        }
    }
    fin.close();

    size_t N = points.size();
    if (N == 0) 
    {
        cerr << "Error: No data points read from file.\n";
        return 1;
    }
    if (K > (int)N) 
    {
        cerr << "Error: num_clusters K is larger than number of data points.\n";
        return 1;
    }

    // Setting number of threads for OpenMP
    omp_set_num_threads(num_threads);

    // Initializing centroids (choose first K points)
    vector<Point> centroids;
    centroids.reserve(K);
    for (int i = 0; i < K; ++i)
        centroids.push_back(points[i]);

    vector<int> assignments(N);
    vector<int> prev_assignments(N, -1);
    double *sum_x = new double[K];
    double *sum_y = new double[K];
    int *count = new int[K];

    bool converged = false;
    int iterations = 0;
    const int max_iter = 1000;
    double start_time = omp_get_wtime();

    while (!converged && iterations < max_iter) 
    {
        iterations++;
        for (int c = 0; c < K; ++c) 
        {
            sum_x[c] = 0.0;
            sum_y[c] = 0.0;
            count[c] = 0;
        }

        if (schedule_type == "static") 
        {
            #pragma omp parallel for schedule(static) default(none) shared(points, centroids, assignments, N, K) reduction(+: sum_x[:K], sum_y[:K], count[:K])
            for (size_t i = 0; i < N; ++i) 
            {
                double px = points[i].x;
                double py = points[i].y;
                int nearest = 0;
                double bestDist = numeric_limits<double>::max();
                for (int c = 0; c < K; ++c) 
                {
                    double dx = px - centroids[c].x;
                    double dy = py - centroids[c].y;
                    double dist = dx*dx + dy*dy;
                    if (dist < bestDist) {
                        bestDist = dist;
                        nearest = c;
                    }
                }
                assignments[i] = nearest;
                sum_x[nearest] += px;
                sum_y[nearest] += py;
                count[nearest] += 1;
            }
        } 
        else 
        {
            #pragma omp parallel for schedule(dynamic) default(none) shared(points, centroids, assignments, N, K) reduction(+: sum_x[:K], sum_y[:K], count[:K])
            for (size_t i = 0; i < N; ++i) 
            {
                double px = points[i].x;
                double py = points[i].y;
                int nearest = 0;
                double bestDist = numeric_limits<double>::max();
                for (int c = 0; c < K; ++c) 
                {
                    double dx = px - centroids[c].x;
                    double dy = py - centroids[c].y;
                    double dist = dx*dx + dy*dy;
                    if (dist < bestDist) 
                    {
                        bestDist = dist;
                        nearest = c;
                    }
                }
                assignments[i] = nearest;
                sum_x[nearest] += px;
                sum_y[nearest] += py;
                count[nearest] += 1;
            }
        }
        
        converged = true;
        for (int c = 0; c < K; ++c) 
        {
            if (count[c] > 0) 
            {
                double new_x = sum_x[c] / count[c];
                double new_y = sum_y[c] / count[c];
                if (fabs(new_x - centroids[c].x) > 1e-6 || fabs(new_y - centroids[c].y) > 1e-6) 
                {
                    converged = false;
                }
                centroids[c].x = new_x;
                centroids[c].y = new_y;
            }
        }

        bool assignment_changed = false;
        for (size_t i = 0; i < N; ++i) 
        {
            if (assignments[i] != prev_assignments[i]) 
            
            {
                assignment_changed = true;
                break;
            }
        }
        if (!assignment_changed) 
        {
            converged = true;
        }
        prev_assignments = assignments;
    }

    double end_time = omp_get_wtime();
    double total_time_ms = (end_time - start_time) * 1000.0;

    cout.setf(std::ios::fixed);
    cout << setprecision(4);
    for (int c = 0; c < K; ++c) 
    {
        cout << "Centroid " << c << ": (" << centroids[c].x << ", " << centroids[c].y << "), ";
        cout << "Size: " << count[c] << "\n";
    }
    cout << "Time: " << fixed << setprecision(4) << total_time_ms << " ms\n";

    delete [] sum_x;
    delete [] sum_y;
    delete [] count;
    return 0;
}
