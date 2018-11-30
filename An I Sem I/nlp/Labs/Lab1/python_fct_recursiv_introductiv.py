import random
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--recursive', action='store_true')

args = parser.parse_args()

Nmic = 0
Nmare = 100
x = 2
nr = random.randint(Nmic, Nmare)

if args.recursive:
  def fc_rec(nr_incercari=0):
    if nr_incercari == x:
      return

    print('Incercari ramase: ', x - nr_incercari)

    ch = int(input('Ghiceste numarul intre ' + str(Nmic) + ' si ' + str(Nmare) + ': '))

    if ch == nr:
      print('Felicitari, ai ghicit numarul!', nr)
      return
    elif ch < nr:
      print('Numarul pe care l-ai ales e mai mic decat numarul cautat.')
    else:
      print('Numarul pe care l-ai ales e mai mare decat numarul cautat.')

    fc_rec(nr_incercari + 1)

  fc_rec()
else:
  nr_incercari = 0
  while (nr_incercari < x):
    print('Incercari ramase: ', x - nr_incercari)
    nr_incercari += 1

    ch = int(input('Ghiceste numarul intre ' + str(Nmic) + ' si ' + str(Nmare) + ': '))

    if ch == nr:
      print('Felicitari, ai ghicit numarul!', nr)
      break
    elif ch < nr:
      print('Numarul pe care l-ai ales e mai mic decat numarul cautat.')
    else:
      print('Numarul pe care l-ai ales e mai mare decat numarul cautat.')
