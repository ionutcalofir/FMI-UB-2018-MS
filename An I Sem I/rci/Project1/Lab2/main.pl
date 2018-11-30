% case
casa(rosie).
casa(alba).
casa(albastra).
casa(galbena).
casa(verde).

% persoane
persoana(britanic).
persoana(suedez).
persoana(danez).
persoana(norvegian).
persoana(german).

% bauturi
bautura(lapte).
bautura(bere).
bautura(ceai).
bautura(apa).
bautura(cafea).

% tigari
tigari(palmall).
tigari(winfield).
tigari(marlboro).
tigari(dunhill).
tigari(rothmans).

% animale
animal(pasare).
animal(caine).
animal(cal).
animal(pisica).
animal(pesti).

% ordine
ord(1).
ord(2).
ord(3).
ord(4).
ord(5).

% utils
diferit(X1, X2) :-
	X1 \= X2.

diferit(X1, X2, X3) :-
	X1 \= X3,
	X2 \= X3.

diferit(X1, X2, X3, X4) :-
	X1 \= X4,
	X2 \= X4,
	X3 \= X4.

diferit(X1, X2, X3, X4, X5) :-
	X1 \= X5,
	X2 \= X5,
	X3 \= X5,
	X4 \= X5.

langa(P1, P2) :-
	Z is abs(P1 - P2),
	Z == 1.

stanga(P1, P2) :-
	Z is P2 - P1,
	Z == 1.

intrebare(V, X, Y, Z, W, P) :-
	puzzle(L),
	member(pers(V, X, Y, Z, W, P), L).

% persoane - casa - bauturi - tigari - animale, numarul casei
puzzle(L) :- 
	L = [
		pers(britanic, X1, Y1, Z1, W1, P1),
		pers(suedez, X2, Y2, Z2, W2, P2),
		pers(danez, X3, Y3, Z3, W3, P3),
		pers(norvegian, X4, Y4, Z4, W4, P4),
		pers(german, X5, Y5, Z5, W5, P5)
	],

	ord(P1), ord(P2), diferit(P1, P2), ord(P3), diferit(P1, P2, P3), ord(P4), diferit(P1, P2, P3, P4), ord(P5), diferit(P1, P2, P3, P4, P5),

	member(pers(britanic, rosie, _, _, _, _), L), % 1

	member(pers(norvegian, _, _, _, _, P_2_norvegian), L), % 2
	member(pers(_, albastra, _, _, _, P_2_albastru), L), % 2
	langa(P_2_norvegian, P_2_albastru), % 2

	member(pers(_, verde, _, _, _, P_3_verde), L), % 3
	member(pers(_, alba, _, _, _, P_3_alba), L), % 3
	stanga(P_3_verde, P_3_alba), % 3

	member(pers(_, verde, cafea, _, _, _), L), % 4

	member(pers(_, _, lapte, _, _, 3), L), % 5

	member(pers(_, galbena, _, dunhill, _, _), L), % 6

	member(pers(norvegian, _, _, _, _, 1), L), % 7

	member(pers(suedez, _, _, _, caine, _), L), % 8

	member(pers(_, _, _, palmall, pasare, _), L), % 9

	member(pers(_, _, _, marlboro, _, P_10_marlboro), L), % 10
	member(pers(_, _, _, _, pisica, P_10_pisica), L), % 10
	langa(P_10_marlboro, P_10_pisica), % 10

	member(pers(_, _, bere, winfield, _, _), L), % 11

	member(pers(_, _, _, dunhill, _, P_12_dunhill), L), % 12
	member(pers(_, _, _, _, cal, P_12_cal), L), % 12
	langa(P_12_dunhill, P_12_cal), % 12

	member(pers(german, _, _, rothmans, _, _), L), % 13

	member(pers(_, _, _, marlboro, _, P_14_marlboro), L), % 14
	member(pers(_, _, apa, _, _, P_14_apa), L), % 14
	langa(P_14_marlboro, P_14_apa), % 14

	casa(X1), casa(X2), diferit(X1, X2), casa(X3), diferit(X1, X2, X3), casa(X4), diferit(X1, X2, X3, X4), casa(X5), diferit(X1, X2, X3, X4, X5),
	bautura(Y1), bautura(Y2), diferit(Y1, Y2), bautura(Y3), diferit(Y1, Y2, Y3), bautura(Y4), diferit(Y1, Y2, Y3, Y4), bautura(Y5), diferit(Y1, Y2, Y3, Y4, Y5),
	tigari(Z1), tigari(Z2), diferit(Z1, Z2), tigari(Z3), diferit(Z1, Z2, Z3), tigari(Z4), diferit(Z1, Z2, Z3, Z4), tigari(Z5), diferit(Z1, Z2, Z3, Z4, Z5),
	animal(W1), animal(W2), diferit(W1, W2), animal(W3), diferit(W1, W2, W3), animal(W4), diferit(W1, W2, W3, W4), animal(W5), diferit(W1, W2, W3, W4, W5).
