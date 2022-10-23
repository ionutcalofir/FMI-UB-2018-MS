#include <bits/stdc++.h>

using namespace std;

ifstream fin("data.in");
ofstream fout("data_dfs.out");

void dfs(int n, vector<int> g[], int s) {
  bool used[n + 1];
  for (int i = 1; i <= n; i++) {
    used[i] = false;
  }
  stack<int> st;

  st.push(s);
  while (!st.empty()) {
    int now = st.top();
    st.pop();

    used[now] = true;
    fout << now << ' ';

    for (int i = g[now].size() - 1; i >= 0; i--) {
      if (used[g[now][i]] == false) {
        st.push(g[now][i]);
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

  dfs(n, g, s);

  return 0;
}
