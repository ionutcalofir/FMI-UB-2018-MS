#include <bits/stdc++.h>

using namespace std;

void print_board(char x0[3][3]) {
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      cout << x0[i][j] << ' ';
    }
    cout << '\n';
  }
}

bool full_board(char x0[3][3]) {
  bool ok = true;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (x0[i][j] == '_') {
        ok = false;
      }
    }
  }

  return ok;
}

int get_score(char x0[3][3]) {
  for (int i = 0; i < 3; i++) {
    if ((x0[0][i] == 'x' && x0[1][i] == 'x' && x0[2][i] == 'x')
        || (x0[i][0] == 'x' && x0[i][1] == 'x' && x0[i][2] == 'x')) {
      return 1;
    }
    if ((x0[0][i] == '0' && x0[1][i] == '0' && x0[2][i] == '0')
        || (x0[i][0] == '0' && x0[i][1] == '0' && x0[i][2] == '0')) {
      return -1;
    }
  }

  if ((x0[0][0] == 'x' && x0[1][1] == 'x' && x0[2][2] == 'x')
      || (x0[0][2] == 'x' && x0[1][1] == 'x' && x0[2][0] == 'x')) {
    return 1;
  }
  if ((x0[0][0] == '0' && x0[1][1] == '0' && x0[2][2] == '0')
      || (x0[0][2] == '0' && x0[1][1] == '0' && x0[2][0] == '0')) {
    return -1;
  }

  return 0;
}

int minmax(char x0[3][3], int depth, bool min_turn, bool max_turn) {
  if (get_score(x0) != 0) {
    return get_score(x0);
  }

  if (full_board(x0)) {
    return get_score(x0);
  }

  if (min_turn) {
    int best_score = 999;

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        if (x0[i][j] == '_') {
          x0[i][j] = '0';
          best_score = min(best_score, minmax(x0, depth + 1, false, true));
          x0[i][j] = '_';
        }
      }
    }

    return best_score;
  } else if (max_turn) {
    int best_score = -999;

    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        if (x0[i][j] == '_') {
          x0[i][j] = 'x';
          best_score = max(best_score, minmax(x0, depth + 1, true, false));
          x0[i][j] = '_';
        }
      }
    }

    return best_score;
  }
}

void solve(char x0[3][3]) {
  int best_score = -999;
  int best_i = -1;
  int best_j = -1;

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      if (x0[i][j] != '_') {
        continue;
      }

      x0[i][j] = 'x';
      int score = minmax(x0, 0, true, false);
      x0[i][j] = '_';

      if (score > best_score) {
        best_score = score;
        best_i = i;
        best_j = j;
      }
    }
  }

  cout << "Best choice: " << best_i << ' ' << best_j << '\n';
  cout << "Score: " << best_score << '\n';
}

int main() {
  char x0[3][3] = {
    {'x', 'x', '0'},
    {'0', '0', '_'},
    {'x', '_', '_'}
  };

  solve(x0);

  return 0;
}
