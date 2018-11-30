def a():
  l = [i for i in range(10) if i % 2 == 1]
  print(l)

def b():
  l = [chr(el) for el in range(ord('a'), ord('z') + 1)]
  print(l, len(l))

def c(n):
  l = [-1 * i if i % 2 == 0 else i for i in range(1, n + 1)]
  print(l)

def d(li):
  l = [el for el in li if el % 2 == 1]
  print(l)

def e(li):
  l = [el for i, el in enumerate(li) if i % 2 == 1]
  print(l)

def f(li):
  l = [el for i, el in enumerate(li) if i % 2 == el % 2]
  print(l)

def g(li):
  l = [(li[i], li[i + 1]) for i in range(0, len(li) - 1)]
  print(l)

def h(n):
  l = [[x * y for x in range(n)] for y in range(n)]
  print(l)

if __name__ == '__main__':
  a()
  b()
  c(10)
  d([1, 2, 3, 4, 5, 6, 7, 8])
  e([1, 2, 3, 4, 5, 6, 7, 8])
  f([2,4,1,7,5,1,8,10])
  g([1, 2, 3, 4])
  h(5)
