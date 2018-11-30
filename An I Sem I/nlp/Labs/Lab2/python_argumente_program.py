from aduna import aduna
import sys

v = []
for i, el in enumerate(sys.argv):
  if i == 0:
    continue

  try:
    nr = int(el)
  except:
    print('Nu se poate face suma.')
    break

  v.append(nr)

s = aduna(*v)
print('Suma este {0}'.format(s))
