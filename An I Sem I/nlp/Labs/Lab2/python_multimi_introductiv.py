def a(l):
  print('Lista e formata doar din elemente egale') if len(set(l)) == 1 else print('Lista nu e formata doar din elemente egale')

def b(s):
  print('Sirul contine toate literele alfabetului') if len(set(s.lower())) == 27 and (True if all([False for c in s.lower() if c >= 'a' and c <= 'z']) else False) else print('Sirul nu contine toate literele alabetului')

def c(a, b):
  print('Sunt anagrame') if sorted(a) == sorted(b) else print('Nu sunt anagrame')

# or with bitmask from 0 to 2^n - 1
def d(v, w, n, li, m):
  if n != -1:
    m.add(frozenset(v[:n + 1]))
    print(v[:n + 1])

  for i, el in enumerate(li):
    if n == -1:
      v[n + 1] = el
      w[n + 1] = i
      d(v, w, n + 1, li, m)
    elif i > w[n]:
      v[n + 1] = el
      w[n + 1] = i
      d(v, w, n + 1, li, m)

def e(a, b):
  for ela in a:
    for elb in b:
      print((ela, elb))

if __name__ == '__main__':
  l = [3, 3, 3, 3]
  a(l)
  print('--')

  b('ana')
  print('--')

  c('bnba', 'bnab')
  print('--')

  l = {1, 2, 'a'}
  m = set()
  d(len(l) * [0], len(l) * [0], -1, l, m)
  print(m)
  print('--')

  e({'a', 'b'}, {'e', 1})
  print('--')
