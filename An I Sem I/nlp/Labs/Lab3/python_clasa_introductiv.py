class Elev():
    nr_elev = 0

    def __init__(self,
                 nume=None,
                 sanatate=None,
                 inteligenta=None,
                 oboseala=None,
                 buna_dispozitie=None):

        if nume == None:
            Elev.nr_elev += 1
            self.nume = 'Necunoscut_' + str(Elev.nr_elev)
            self.sanatate = 0
            self.inteligenta = 0
            self.oboseala = 0
            self.buna_dispozitie = 0
        else:
            self.nume = nume
            self.sanatate = sanatate
            self.inteligenta = inteligenta
            self.oboseala = oboseala
            self.buna_dispozitie = buna_dispozitie

class Activitati():
    def __init__(self,
                 nume,
                 factor_sanatate,
                 factor_inteligenta,
                 factor_oboseala,
                 factor_dispozitie,
                 durata):
        self.nume = nume
        self.factor_sanatate = factor_sanatate
        self.factor_inteligenta = factor_inteligenta
        self.factor_oboseala = factor_oboseala
        self.factor_dispozitie = factor_dispozitie
        self.durata = durata

if __name__ == '__main__':
    ob1 = Elev()
    print(ob1.nume)
    ob2 = Elev()
    print(ob2.nume)
