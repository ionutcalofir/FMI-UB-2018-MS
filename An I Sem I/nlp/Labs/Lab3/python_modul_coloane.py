import afisare

try:
    afisare.col2([('abc', 10), (4, 'xyz')], 6)
    afisare.col3([('abc', 10, 3), ('aa', 'bb', 'cc')], 6)

    # afisare.col2([('aaaaaaaaa', 10), (4, 'xyz')], 6) # 1
    # afisare.col2([('aa', 10, 3), (4, 'xyz')], 6) 2
    # afisare.col2([('aa', 10, 3), (4, 'xyz')], 'test') # 3
except afisare.ColumnDimError as e:
    print(e.message)
except afisare.ListArgumentError as e:
    print(e.message)
except afisare.GenericArgumentError as e:
    print(e.message)
else:
    print('everything went fine!')
finally:
    print('the end!')
