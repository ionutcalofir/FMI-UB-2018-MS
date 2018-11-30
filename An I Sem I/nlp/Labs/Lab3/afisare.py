class ColumnDimError(Exception):
    def __init__(self, message):
        self.message = message

class ListArgumentError(Exception):
    def __init__(self, message):
        self.message = message

class GenericArgumentError(Exception):
    def __init__(self, message):
        self.message = message

caracter_de_bordare = '*'

def col2(li, dim):
    if not isinstance(li, list) or not isinstance(dim, int):
        raise GenericArgumentError('Parametrii functiei nu sunt valizi!')

    for el in li:
        if not isinstance(el, tuple) or len(el) != 2:
            raise ListArgumentError('Tuplul nu este valid!')

        el1, el2 = el

        str_el1 = str(el1)
        str_el2 = str(el2)

        if len(str_el1) > dim or len(str_el2) > dim:
            raise ColumnDimError('Un element din tuplu este mai mare decat coloana!')

        s = str_el1 + caracter_de_bordare * (dim - len(str_el1)) + ' ' \
            + str_el2 + caracter_de_bordare * (dim - len(str_el2))

        print(s)

def col3(li, dim):
    if not isinstance(li, list) or not isinstance(dim, int):
        raise GenericArgumentError('Parametrii functiei nu sunt valizi!')

    for el in li:
        if not isinstance(el, tuple) or len(el) != 3:
            raise ListArgumentError('Tuplul nu este valid!')

        el1, el2, el3 = el

        str_el1 = str(el1)
        str_el2 = str(el2)
        str_el3 = str(el3)

        if len(str_el1) > dim or len(str_el2) > dim or len(str_el3) > dim:
            raise ColumnDimError('Un element din tuplu este mai mare decat coloana!')

        s = str_el1.center(dim, caracter_de_bordare) + '|' \
            + str_el2.center(dim, caracter_de_bordare) + '|' \
            + str_el3.center(dim, caracter_de_bordare)

        print(s)

if __name__ == '__main__':
    col2([('abc', 10), (4, 'xyz')], 6)
    col3([('abc', 10, 3), ('aa', 'bb', 'cc')], 6)
