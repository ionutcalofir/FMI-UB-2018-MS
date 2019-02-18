main_solve :-
	open('lab4.txt', read, Fl_read),
	read(Fl_read, Clauses),
	read(Fl_read, Elems),
	close(Fl_read),
	(terminat_multimea_vida(Elems) -> write('DA'), fail;
		reverse(Clauses, Clauses_rev, []),
		solve(Elems, Clauses_rev, R, Rd, Elems, Clauses),
		(terminat_multimea_vida(Rd) -> write('NU');
			del_first(Elems, Elems_new),
			reverse(R, R_rev, []),
			append(R_rev, Elems_new, Elems_final),
			open('lab4.txt', write, Fl_write),
			write(Fl_write, Clauses_rev),
			write(Fl_write, '.'),
			write(Fl_write, '\n'),
			write(Fl_write, Elems_final),
			write(Fl_write, '.'),
			close(Fl_write),
			main_solve)).

del_first([_ | T], T).

terminat_multimea_vida([]).

del_elems([], _, []).
del_elems([H_c | T_c], Rd, L) :-
	del_elems(T_c, Rd, L1),
	(\+ member(H_c, Rd) ->
				append(L1, [H_c], L2),
				L = L2;
			append(L1, [], L2),
			L = L2).

check_clause([], _).
check_clause([Elem | T], Elem) :-
	check_clause(T, Elem).
check_clause([H | T], Elem) :-
	(check_not(Elem) ->
			\+ check_not(H),
			check_clause(T, Elem);
		check_not(H),
		check_clause(T, Elem)).

reverse_l([], []).
reverse_l([H | T], R) :-
	reverse_l(T, R1),
	(check_not(H) ->
			get_not_term(H, H_new),
			append(R1, [H_new], R2),
			R = R2;
		append(R1, [n(H)], R2),
		R = R2).

reverse([], Z, Z).
reverse([H | T], Z, Acc) :- reverse(T, Z, [H | Acc]).

solve([H_elem | _], Clauses, L, Ld, Elems, Cl) :-
	add_new_clause(H_elem, Clauses, R, Rd, Elems, Cl),
	L = R,
	Ld = Rd.


add_new_clause(_, [], [], [], _, _).
add_new_clause(Elem, [H_clause | T_clause], L, Ld, Elems, Clauses) :-
	add_new_clause(Elem, T_clause, L1, Ld1, Elems, Clauses),
	(check_clause(H_clause, Elem), member(Elem, H_clause) ->
				delete(H_clause, Elem, LL),
				reverse_l(LL, LL_rev),
				append(L1, LL_rev, L2),
				append(Ld1, [Elem], Ld2),
				L = L2,
				Ld = Ld2,
			
				del_first(Elems, Elems_new),
				reverse(L2, R_rev, []),
				append(R_rev, Elems_new, Elems_final),
				open('lab4.txt', write, Fl_write),
				write(Fl_write, Clauses),
				write(Fl_write, '.'),
				write(Fl_write, '\n'),
				write(Fl_write, Elems_final),
				write(Fl_write, '.'),
				close(Fl_write),
				main_solve;

			append(L1, [], L2),
			append(Ld1, [], Ld2),
			L = L2,
			Ld = Ld2).

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
