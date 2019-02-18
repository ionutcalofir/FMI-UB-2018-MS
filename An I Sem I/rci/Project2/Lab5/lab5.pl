:- (dynamic cube/3).
:- (dynamic counter/1).

main_solve :-
	read_file_to_terms('lab5_kb.txt', Terms, []),
	add_p(Terms),
	/*
	*open('lab5_rules.txt', read, Fl_read),
	*read(Fl_read, Rule11),
	*close(Fl_read).
	*/
	solve.

solve :-
	rule1, !.
solve :-
	rule2, !.
solve :-
	findall(cube(Name, Size, Position),
		cube(Name, Size, Position),
		L),
	findall(counter(V),
		counter(V),
		Lc),
	print(Lc), nl,
	print(L).

is_size_max(Max_size) :-
	findall(cube(Name, Size, heap),
		cube(Name, Size, heap),
		L),
	solve_is_max(Max_size, L).
solve_is_max(_, []).
solve_is_max(Max_size, [cube(_, Size, _) | T]) :-
	Max_size >= Size,
	solve_is_max(Max_size, T).

is_hand :-
	cube(_, _, hand).

rule1 :-
	cube(Name, Size, heap),
	is_size_max(Size),
	\+ is_hand,
	retract(cube(Name, Size, heap)),
	assertz(cube(Name, Size, hand)),
	solve, !.

rule2 :-
	cube(Name, Size, hand),
	counter(V),
	retract(cube(Name, Size, hand)),
	assertz(cube(Name, Size, V)),
	retract(counter(V)),
	V_new is V + 1,
	assertz(counter(V_new)),
	solve, !.

add_p([]).
add_p([H | T]) :-
	assertz(H),
	add_p(T).
