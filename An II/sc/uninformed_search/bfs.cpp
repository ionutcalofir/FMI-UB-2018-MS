#include <bits/stdc++.h>

using namespace std;

ifstream fin("data.in");
ofstream fout("data_bfs.out");

void bfs(int n, vector<int> g[], int s) {
  bool used[n + 1];
  for (int i = 1; i <= n; i++) {
    used[i] = false;
  }
  queue<int> q;

  q.push(s);
  while (!q.empty()) {
    int now = q.front();
    q.pop();

    used[now] = true;
    fout << now << ' ';

    for (int i = 0; i < g[now].size(); i++) {
      if (used[g[now][i]] == false) {
        q.push(g[now][i]);
      }
    }
  }
}

int main() {
  int n, m , s;
  fin >> n >> m >> s;
  vector<int> g[n + 1];
  for (int i = 0, nod1, nod2; i < m; i++) {
    fin >> nod1 >> nod2;
    g[nod1].push_back(nod2);
  }

  bfs(n, g, s);

  return 0;
}
