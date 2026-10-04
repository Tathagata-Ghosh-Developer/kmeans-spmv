#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

struct Point 
{
    double x;
    double y;
};

int main(int argc, char** argv) 
{
    if (argc < 3) 
    {
        cerr << "Usage: " << argv[0] << " <input_file> <num_clusters>\n";
        return 1;
    }
    string input_file = argv[1];
    int K;
    try 
    {
        K = stoi(argv[2]);
    } 
    catch (exception &e) 
    {
        cerr << "Error: num_clusters must be an integer\n";
        return 1;
    }
    if (K <= 0) 
    {
        cerr << "Error: num_clusters must be positive\n";
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

    // Initializing centroids (choosing first K points as initial centroids)
    vector<Point> centroids;
    centroids.reserve(K);
    for (int i = 0; i < K; ++i) {
        centroids.push_back(points[i]);
    }

    vector<int> assignments(N);
    vector<int> prev_assignments(N, -1);
    vector<double> sum_x(K);
    vector<double> sum_y(K);
    vector<int> count(K);

    bool converged = false;
    int iterations = 0;
    const int max_iter = 1000;
    double start_time = omp_get_wtime();

    while (!converged && iterations < max_iter)
    {
        iterations++;
        fill(sum_x.begin(), sum_x.end(), 0.0);
        fill(sum_y.begin(), sum_y.end(), 0.0);
        fill(count.begin(), count.end(), 0);

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
            break;
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

    return 0;
}
