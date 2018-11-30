import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--a', action='store_true')
parser.add_argument('--b', action='store_true')
parser.add_argument('--c', action='store_true')

args = parser.parse_args()

s = input('Introduceti sirul: ')

if args.a:
  for i in range(0, len(s)):
    print(s[i:] + s[:i])
elif args.b:
  for i in range(0, len(s)):
    print(s[-i:] + s[:-i])
elif args.c:
  for i in range(1, int(len(s) / 2) + 1):
    print(s[:i] + '|' + s[-i:])
