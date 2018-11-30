def a(reuniune, intersectat, dif):
  A = dif | intersectat
  # B = (reuniune - A) | intersectat
  B  = reuniune - dif
  return (A, B)

def b(intersectat, AB, BA):
  A = AB | intersectat
  B = BA | intersectat
  return (A, B)

if __name__ == '__main__':
  A = {1, 2, 3, 4, 5}
  B = {3, 4, 5}
  (newA, newB) = a(A | B, A & B, A - B)
  print(newA, newB)

  (newA, newB) = b(A & B, A - B, B - A)
  print(newA, newB)
