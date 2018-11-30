class Dreptunghi:
    """O clasa care memoreaza un dreptunghi format din caractere"""
    #atribute de clasa (statice)
    liniiSeparatoare=True;
    simbolContur="#"

    def __init__(self, inaltime, latime, simbolContinut, simbolSeparator="-"):
        #atribute de instanta
        self.inaltime = inaltime
        self.latime = latime
        self.simbolContinut = simbolContinut

    @classmethod
    def afisSimbolContur(cls):
        print('Simbol contur: '+ cls.simbolContur)

    def afisProprietatiInstanta(self):
        for (k,v) in self.__dict__.items() :
            print("{} = {}".format(k,v))

    @classmethod
    def afisDoc(cls):
        print('Informatii clasa:\n'+ cls.__doc__)

    @classmethod
    def afisNumeClasa(cls):
        print('Nume clasa:\n'+ cls.__name__)

    def __str__(self):
        sir="Prop clasa:\n"
        for (k,v) in self.__class__.__dict__.items() :
            sir+="{} = {}\n".format(k,v)
        sir="Prop instanta:\n"
        for (k,v) in self.__dict__.items() :
            sir+="{} = {}\n".format(k,v)
        return(sir)


    def __repr__(self):
        sir=self.simbolContur*self.latime+"\n";
        for i in range(self.inaltime-2):
            sir+=self.simbolContur+self.simbolContinut*(self.latime-2)+self.simbolContur+"\n"
        sir+=self.simbolContur*self.latime+"\n"
        return(sir) 

if __name__ == '__main__':
    ob = Dreptunghi(5, 5, '*')
    print(repr(ob))
