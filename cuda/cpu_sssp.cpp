#include <stdio.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>
#include <limits>
#include <string>
#include <ctime>
#include <algorithm>
#include <sstream>
using namespace std;
using std::cout;
using std::endl;
using namespace std::chrono;

void load_data(const char *f, std::vector<int> &vec){
    std::ifstream inputfile;
    int n;
    inputfile.open(f);
    while(inputfile.ignore() && (inputfile >> n)){vec.push_back(n);}
    inputfile.close();
}

void fix_INDEX_edges(std::vector<int> &V, std::vector<int> &E, int l, int r){
    for (int index = 0; index < E.size(); index++) {
        l=0; r=V.size()-1;
        while (l <= r) {
            int m = l + (r - l) / 2;
            if (V[m] == E[index]) {
                E[index] = m;
                break;
            }
            if (V[m] < E[index]) {l = m + 1;} else {r = m - 1;}
        }
    }
}

int main(){
    std::vector<int> V, I, E, W;
    load_data("/home/styagi/rand_1000.gr_V.csv", V);
    load_data("/home/styagi/rand_1000.gr_I.csv", I);
    load_data("/home/styagi/rand_1000.gr_E.csv", E);
    load_data("/home/styagi/rand_1000.gr_W.csv", W);

    std::vector<int> P(V.size(), std::numeric_limits<int>::max());
    std::vector<int> Q(V.size(), -1);

    fix_INDEX_edges(V, E, 0, V.size()-1);

    P[0] = 0;
    Q[0] = 0;

    auto start_time = high_resolution_clock::now();
    for (int i = 0; i < V.size(); ++i) {
        for (int j = 0; j < I.size()-1; ++j) {
            for (int k = 0; k < I[j+1]; ++k) {
                int p, q, r, dp, dq;
                p = V[j];
                q = V[E[k]];
                r = W[k];
                dp = P[j];
                dq = P[E[k]];
                if (dp + r < dq){
                    P[E[k]] = dp + r;
                    Q[E[k]] = p;
                }

            }
        }
    }
    auto end_time = high_resolution_clock::now();
    auto exec_time = duration_cast<microseconds>(end_time - start_time);
    cout << exec_time.count() << endl;
}