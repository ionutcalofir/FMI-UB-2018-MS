:- dynamic pred/1.

main_solve :-
	open('data.txt', read, Fl_read),
	read(Fl_read, Clauses),
	close(Fl_read),
	assert_term_clause(Clauses),
	solve_write(Clauses, []).

assert_term_clause([]).
assert_term_clause([H_clause | T_clause]) :-
	assert_term(H_clause),
	assert_term_clause(T_clause).

assert_term([]).
assert_term([H_term | T_term]) :-
	(check_not(H_term) ->
			get_not_term(H_term, H_term_new);
		H_term_new = n(H_term)),
	(not(pred(H_term)) -> assertz(pred(H_term)); true),
	(not(pred(H_term_new)) -> assertz(pred(H_term_new)); true),
	assert_term(T_term).

solve_write(Clauses, L) :-
	(solve(Clauses, L) ->
			write('DA'), nl;
		write('NU'), nl).

solve([], L) :- write(L), nl.
solve(Clauses, L_c) :-
	pred(X),
	not(member(X, L_c)),
	(check_not(X) ->
			get_not_term(X, XX),
			not(member(XX, L_c));
		not(member(n(X), L_c))),
	new_clauses(Clauses, X, L),
	not(member([], L)),
	append(L_c, [X], L1),
	solve(L, L1), !.

new_clauses([], _, []).
new_clauses([H_clause | T_clause], X, L) :-
	(check_not(X) -> % Ex: n(t)
			get_not_term(X, XX),% Ex: n(t) -> t
			(member(X, H_clause) ->
					new_clauses(T_clause, X, L);
				(member(XX, H_clause) ->
						delete(H_clause, XX, H_new),
						L = [H_new | T],
						new_clauses(T_clause, X, T);
					L = [H_clause| T],
					new_clauses(T_clause, X, T)));
		(member(X, H_clause) ->
				new_clauses(T_clause, X, L);
			(member(n(X), H_clause) ->
					delete(H_clause, n(X), H_new),
					L = [H_new | T],
					new_clauses(T_clause, X, T);
				L = [H_clause | T],
				new_clauses(T_clause, X, T)))).

% Remove the not from the term. Ex: n(p) -> p
%-------------------------------------------------------------------------------
get_not_term(T, X) :-
	term_to_atom(T, A),
	atom_chars(A, C),
	solve_get_not_term(C, X).

solve_get_not_term([_, _ | C_T], X) :-
	remove_last_elem(C_T, Y),
	atom_chars(X, Y).

remove_last_elem([_], []).
remove_last_elem([X | Xs], [X | Xnew]) :-
	remove_last_elem(Xs, Xnew).
%-------------------------------------------------------------------------------

% Check if a term has not. Ex: checknot(n(p)) -> true
%-------------------------------------------------------------------------------
check_not(X) :-
	term_to_atom(X, B),
	atom_chars(B, A),
	is_not(A).

is_not(X) :-
	verify_start(X),
	verify_end(X).

verify_start([X1, X2 | _]) :-
	X1 == 'n',
	X2 == '('.

verify_end([X | []]) :-
	X == ')'.
verify_end([_ | T]) :-
	verify_end(T).
%-------------------------------------------------------------------------------
