import sys

# 0 1 2 3
# 0 1 2 3
# 0 1 2 3
# 0 1 2 3
n = 4 # row
mat = [[el for el in range (0, n)] for x in range(0, n)]

def prima_diag():
  for i in range(0, n):
    print(mat[i][i])

def sec_diag():
  for i in range(n, 0, -1):
    print(i)

def contur():
  for j in range(0, n):
    print(mat[0][j])
  for i in range(1, n):
    print(mat[i][n - 1])
  for j in range(n - 2, 0, -1):
    print(mat[n - 1][j])
  for i in range(n - 1, 0, -1):
    print(mat[i][0])

def suma():
  s = 0
  for i in range(0, n):
    for j in range(0, n):
      s += mat[i][j]

  print(s)

def iesire():
  sys.exit()

meniu = {'1': prima_diag,
         '2': sec_diag,
         '3': contur,
         '4': suma,
         '5': iesire}

while True:
  op = input('Alegeti optiunea: ')

  if op not in meniu:
    continue

  meniu[op]()
