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
    inputfile.close()
}

void fix_INDEX_edges(std::vector<int> &A, std::vector<int> &C, int l, int r){
    for (int index = 0; index < E.size(); index++) {
        l=0; r=V.size()-1;
        while (l <= r) {
            int m = l + (r - l) / 2;
            if (A[m] == C[index]) {
                C[index] = m;
                break;
            }
            if (A[m] < C[index]) {l = m + 1;} else {r = m - 1;}
        }
    }
}

int main(){
    std::vector<int> A, B, C, D;
    load_data(("/home/styagi/rand_1000.gr_V.csv").c_str(), A);
    load_data(("/home/styagi/rand_1000.gr_I.csv").c_str(), B);
    load_data(("/home/styagi/rand_1000.gr_E.csv").c_str(), C);
    load_data(("/home/styagi/rand_1000.gr_W.csv").c_str(), D);

    std::vector<int> P(A.size(), std::numeric_limits<int>::max());
    std::vector<int> Q(A.size(), -1);

    fix_INDEX_edges(A, C, 0, A.size()-1);

    P[0] = 0;
    Q[0] = 0;

    auto start_time = high_resolution_clock::now();
    for (int i = 0; i < A.size(); ++i) {
        for (int j = 0; j < B.size()-1; ++j) {
            for (int k = 0; k < B[j+1]; ++k) {
                int p, q, r, dp, dq;
                p = A[j];
                q = A[C[k]];
                r = D[k];
                dp = P[j];
                dq = P[C[k]];
                if (dp + r < dq){
                    P[C[k]] = dp + r;
                    Q[C[k]] = p;
                }

            }
        }
    }
    auto end_time = high_resolution_clock::now();
    auto exec_time = duration_cast<microseconds>(end_time - start_time);
    cout << exec_time.count() << endl;
}