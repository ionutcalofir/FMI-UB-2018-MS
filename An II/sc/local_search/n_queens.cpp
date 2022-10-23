#include <bits/stdc++.h>

using namespace std;

const int N = 4;

struct queen {
  int i, j;
};

queen Q[N + 1];

void generate_state() {
  for (int k = 1; k <= N; k++) {
    Q[k].i = 1;
    Q[k].j = k;
  }

  for (int k = 1; k <= N; k++) {
    Q[k].i = rand() % N + 1;
  }
}

int get_score(bool verbose=false) {
  if (verbose) {
    cout << "Score pairs: \n";
  }
  int table_score[N + 1][N + 1];

  for (int i = 1; i <= N; i++) {
    for (int j = 1; j <= N; j++) {
      table_score[i][j] = 0;
    }
  }
  for (int k = 1; k <= N; k++) {
    table_score[Q[k].i][Q[k].j] = 1;
  }

  int score = 0;
  for (int k = 1; k <= N; k++) {
    int inow = Q[k].i;
    int jnow = Q[k].j;
    table_score[inow][jnow] = 0;

    /*
     * down
     */
    for (int pos = 1; pos <= N; pos++) {
      int i = inow + pos;
      int j = jnow;

      if (i < 1 || j < 1 || i > N || j > N) {
        break;
      }

      if (table_score[i][j] == 1) {
        score += 1;
        if (verbose) {
          cout <<  inow << ' ' << jnow << " - " << i << ' ' << j << '\n';
        }
      }
    }

    /*
     * up
     */
    for (int pos = 1; pos <= N; pos++) {
      int i = inow - pos;
      int j = jnow;

      if (i < 1 || j < 1 || i > N || j > N) {
        break;
      }

      if (table_score[i][j] == 1) {
        score += 1;
        if (verbose) {
          cout <<  inow << ' ' << jnow << " - " << i << ' ' << j << '\n';
        }
      }
    }

    /*
     * right
     */
    for (int pos = 1; pos <= N; pos++) {
      int i = inow;
      int j = jnow + pos;

      if (i < 1 || j < 1 || i > N || j > N) {
        break;
      }

      if (table_score[i][j] == 1) {
        score += 1;
        if (verbose) {
          cout <<  inow << ' ' << jnow << " - " << i << ' ' << j << '\n';
        }
      }
    }

    /*
     * left
     */
    for (int pos = 1; pos <= N; pos++) {
      int i = inow;
      int j = jnow - pos;

      if (i < 1 || j < 1 || i > N || j > N) {
        break;
      }

      if (table_score[i][j] == 1) {
        score += 1;
        if (verbose) {
          cout <<  inow << ' ' << jnow << " - " << i << ' ' << j << '\n';
        }
      }
    }

    /*
     * up - right
     */
    for (int pos = 1; pos <= N; pos++) {
      int i = inow - pos;
      int j = jnow + pos;

      if (i < 1 || j < 1 || i > N || j > N) {
        break;
      }

      if (table_score[i][j] == 1) {
        score += 1;
        if (verbose) {
          cout <<  inow << ' ' << jnow << " - " << i << ' ' << j << '\n';
        }
      }
    }

    /*
     * up - left
     */
    for (int pos = 1; pos <= N; pos++) {
      int i = inow - pos;
      int j = jnow - pos;

      if (i < 1 || j < 1 || i > N || j > N) {
        break;
      }

      if (table_score[i][j] == 1) {
        score += 1;
        if (verbose) {
          cout <<  inow << ' ' << jnow << " - " << i << ' ' << j << '\n';
        }
      }
    }

    /*
     * down - right
     */
    for (int pos = 1; pos <= N; pos++) {
      int i = inow + pos;
      int j = jnow + pos;

      if (i < 1 || j < 1 || i > N || j > N) {
        break;
      }

      if (table_score[i][j] == 1) {
        score += 1;
        if (verbose) {
          cout <<  inow << ' ' << jnow << " - " << i << ' ' << j << '\n';
        }
      }
    }

    /*
     * down - left
     */
    for (int pos = 1; pos <= N; pos++) {
      int i = inow + pos;
      int j = jnow - pos;

      if (i < 1 || j < 1 || i > N || j > N) {
        break;
      }

      if (table_score[i][j] == 1) {
        score += 1;
        if (verbose) {
          cout <<  inow << ' ' << jnow << " - " << i << ' ' << j << '\n';
        }
      }
    }
  }

  return score;
}

void print_table() {
  int table_print[N + 1][N + 1];

  for (int i = 1; i <= N; i++) {
    for (int j = 1; j <= N; j++) {
      table_print[i][j] = 0;
    }
  }
  for (int k = 1; k <= N; k++) {
    table_print[Q[k].i][Q[k].j] = 1;
  }

  for (int i = 1; i <= N; i++) {
    for (int j = 1; j <= N; j++) {
      cout << table_print[i][j] << ' ';
    }
    cout << '\n';
  }
}

void solve() {
  while (true) {
    int current_score = get_score();
    bool better_score = false;

    for (int k = 1; k <= N; k++) {
      int current_i = Q[k].i;
      int current_j = Q[k].j;

      for (int ii = 1; ii <= N; ii++) {
        Q[k].i = ii;
        int now_score = get_score();

        if (now_score < current_score) {
          better_score = true;
        }

        if (better_score) {
          break;
        }
      }

      if (better_score) {
        break;
      } else {
        Q[k].i = current_i;
      }
    }

    if (!better_score) {
      break;
    }
  }
}

int main() {
  srand (time(NULL));

  generate_state();
  solve();

  int score = get_score(true);
  cout << "Best score found: " << score << '\n';
  print_table();

  return 0;
}
