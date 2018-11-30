li = ["haha", "poc", "Poc", "POC", "haHA", "hei", "hey", "HahA", "poc", "Hei"]
d = {}

for el in li:
  if el.lower() not in d.keys():
    d[el.lower()] = 1
  else:
    d[el.lower()] += 1

for el in li:
  print('Elementul ' + el + ' apare de ' + str(d[el.lower()]) + ' ori.')
