main_solve :-
	open('lab4.txt', read, Fl_read),
	read(Fl_read, Clauses),
	read(Fl_read, Elems),
	read(Fl_read, Rezolvat),
	close(Fl_read),
	(terminat_bine(Elems, Rezolvat) -> print('DA');
			solve(Clauses, Rezolvat, R, Rc),
			(terminat_multimea_vida(R) -> print('NU');
					dif(Clauses, Rc, Clauses_new),
					append(Rezolvat, R, R_new),
					open('lab4.txt', write, Fl_write),
					write(Fl_write, Clauses_new),
					write(Fl_write, '.'),
					write(Fl_write, '\n'),
					write(Fl_write, Elems),
					write(Fl_write, '.'),
					write(Fl_write, '\n'),
					write(Fl_write, R_new),
					write(Fl_write, '.'),
					close(Fl_write),
					main_solve)).

dif([], _, []).
dif([H | T], Rc, L) :-
	dif(T, Rc, L1),
	(member(H, Rc) ->
			append(L1, [], L2),
			L = L2;
		append(L1, [H], L2),
		L = L2).

terminat_multimea_vida([]).

terminat_bine([], _).
terminat_bine([H_elem | T_elem], Rezolvat) :-
	member(H_elem, Rezolvat),
	terminat_bine(T_elem, Rezolvat).

rezolvat_bine([], _).
rezolvat_bine([H_elem | T_elem], Rezolvat) :-
	(check_not(H_elem) ->
			get_not_term(H_elem, H_elem_new),
			member(H_elem_new, Rezolvat),
			rezolvat_bine(T_elem, Rezolvat);
		member(n(H_elem), Rezolvat),
		rezolvat_bine(T_elem, Rezolvat)).

count_neg([], 0).
count_neg([H | T], N) :-
	count_neg(T, N1),
	(check_not(H) ->
			N is N1 + 1;
		N is N1).

count_pos([], 0).
count_pos([H | T], N) :-
	count_pos(T, N1),
	(check_not(H) ->
			N is N1;
		N is N1 + 1).

get_neg_terms([], []).
get_neg_terms([H | T], L) :-
	get_neg_terms(T, L1),
	(check_not(H) ->
			append(L1, [H], L2),
			L = L2;
		L = L1).

get_pos_terms([], []).
get_pos_terms([H | T], L) :-
	get_pos_terms(T, L1),
	(check_not(H) ->
			L = L1;
		append(L1, [H], L2),
		L = L2).

get_term([H | _], T) :-
	T = H.

check_clause(Clause, Rezolvat, El) :-
	count_neg(Clause, N_neg),
	count_pos(Clause, N_pos),
	N_neg >= 0,
	N_pos == 1,
	get_neg_terms(Clause, Terms_neg),
	rezolvat_bine(Terms_neg, Rezolvat),
	get_pos_terms(Clause, Terms_pos),
	get_term(Terms_pos, Term),
	El = Term.

/*
 *check_clause(Clause, Rezolvat, El) :-
 *        count_neg(Clause, N_neg),
 *        count_pos(Clause, N_pos),
 *        N_neg == 1,
 *        N_pos >= 0,
 *        get_neg_terms(Clause, Terms_neg),
 *        get_pos_terms(Clause, Terms_pos),
 *        rezolvat_bine(Terms_pos, Rezolvat),
 *        get_term(Terms_neg, Term),
 *        El = Term.
 */

solve([], _, [], []).
solve([H_clause | T_clause], Rezolvat, L, Lc) :-
	solve(T_clause, Rezolvat, L1, Lc1),
	(check_clause(H_clause, Rezolvat, At) ->
				append(L1, [At], L2),
				append(Lc1, [H_clause], Lc2),
				Lc = Lc2,
				L = L2;
			L = L1,
			Lc = Lc1).

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
