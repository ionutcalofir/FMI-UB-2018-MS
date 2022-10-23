#include <bits/stdc++.h>

using namespace std;

ifstream fin("data.in");
ofstream fout("data.out");

const int NMAX = 100;
const int MMAX = 100;

const int KMAX = 5;
const int dx[5] = {-1, 0, 1, 0};
const int dy[5] = {0, 1, 0, -1};

struct ComparePQ {
  bool operator() (pair<int, pair<int, pair<int, int>>> A,
                   pair<int, pair<int, pair<int, int>>> B) {
    if (A.first == B.first){
      return A.second.first > B.second.first;
    }

    return A.first > B.first;
  }
};

int manhattan_distance(int i, int j, int ei, int ej) {
  return abs(ei - i) + abs(ej - j);
}

/*
 * pq(total_distance, (distance_to_start, (i, j)))
 */
void a_star(int n, int m, int si, int sj, int ei, int ej, int h[NMAX][MMAX]) {
  priority_queue<pair<int, pair<int, pair<int, int>>>,
                 vector<pair<int, pair<int, pair<int, int>>>>,
                 ComparePQ> pq;

  bool used[NMAX][MMAX];
  for (int i = 0; i < NMAX; i++) {
    for (int j = 0; j < MMAX; j++) {
      used[i][j] = false;
    }
  }

  pq.push(make_pair(0, make_pair(0, make_pair(si, sj))));
  while (!pq.empty()) {
    pair<int, pair<int, pair<int, int>>> pq_top = pq.top();
    pq.pop();

    int dist_to_end = pq_top.first;
    int dist_to_start = pq_top.second.first;
    int i = pq_top.second.second.first;
    int j = pq_top.second.second.second;

    if (used[i][j] == true) {
      continue;
    }
    used[i][j] = true;

    fout << i << ' ' << j << " - " << dist_to_start << ' ' << dist_to_end << '\n';

    if (i == ei && j == ej) {
      break;
    }

    for (int k = 0; k < KMAX; k++) {
      int inow = i + dx[k];
      int jnow = j + dy[k];

      if (inow < 1 || jnow < 1 || inow > n || jnow > m) {
        continue;
      }

      if (h[inow][jnow] == 1) {
        continue;
      }

      int dist_to_start_now = dist_to_start + 1;
      int dist_to_end_now = manhattan_distance(inow, jnow, ei, ej);
      int total_dist = dist_to_start_now + dist_to_end_now;
      pq.push(make_pair(total_dist, make_pair(dist_to_start_now, make_pair(inow, jnow))));
    }
  }
}

int main() {
  int n, m, si, sj, ei, ej;
  fin >> n >> m;
  fin >> si >> sj;
  fin >> ei >> ej;

  int h[NMAX][MMAX];

  for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= m; j++) {
      fin >> h[i][j];
    }
  }

  a_star(n, m, si, sj, ei, ej, h);

  return 0;
}
