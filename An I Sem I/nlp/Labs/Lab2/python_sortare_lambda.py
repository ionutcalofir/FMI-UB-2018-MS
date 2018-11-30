def a(li):
  return sorted(li, key=lambda x: str(x))

def b(li):
  return sorted(li, key=lambda x: str(x)[::-1])

def c(li):
  return sorted(li, key=lambda x: len(str(x)))

def d(li):
  return sorted(li, key=lambda x: len(set(str(x))))

def e(li):
  return sorted(li, key=lambda x: eval(x))

if __name__ == '__main__':
  res = a([123, 102])
  print(res)

  res = b([520, 125])
  print(res)

  res = c([109, 52, 49])
  print(res)

  res = d([109, 52, 49, 11111111])
  print(res)

  res = e(['1+2+3', '2-5', '3+4', '5*10'])
  print(res)
