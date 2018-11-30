main_solve :-
	open('data.txt', read, Fl_read),
	read(Fl_read, Clauses),
	close(Fl_read),
	(terminat_clauza_vida(Clauses) -> write('NU POATE FI SATISFACUT');
		solve(Clauses, R),
		sort(R, R2), % remove duplicates
		append(R2, Clauses, Clauses2),
		sort(Clauses2, Clauses_final),
		(Clauses_final == Clauses ->
				write('SATISFACUT'); % daca R nu e diferit de Clauses, atunci multimea e satisfacuta
			open('data.txt', write, Fl_write),
			write(Fl_write, Clauses_final),
			write(Fl_write, '.'),
			close(Fl_write),
			main_solve)).

% Check if [] is reached.
terminat_clauza_vida(Clauses) :-
	member([], Clauses).

solve([], []).
solve([H_clause | T_clause], L) :- % need to construct the new list of clauses
	solve(T_clause, L1),
	combine_clause_with_rest(H_clause, H_clause, T_clause, R),
	append(L1, R, L2),
	L = L2.

combine_clause_with_rest(_, [], _, []).
combine_clause_with_rest(H_clause, [H_elem | T_elem], T_clause, L) :-
	combine_clause_with_rest(H_clause, T_elem, T_clause, L1),
	check_elem_in_clauses(H_clause, H_elem, T_clause, R),
	append(L1, R, L2),
	L = L2.

check_elem_in_clauses(_, _, [], []).
check_elem_in_clauses(Cl, El, [H | T], L) :-
	(check_not(El) ->
			get_not_term(El, El_new),
			(member(El_new, H) ->
					delete(H, El_new, H_new),
					delete(Cl, El, Cl_new),
					append(H_new, Cl_new, HH_new),
					sort(HH_new, HH), % remove duplicates
					L = [HH | TT],
					check_elem_in_clauses(Cl, El, T, TT);
				check_elem_in_clauses(Cl, El, T, L));
		(member(n(El), H) ->
				delete(H, n(El), H_new),
				delete(Cl, El, Cl_new),
				append(H_new, Cl_new, HH_new),
				sort(HH_new, HH), % remove duplicates
				L = [HH | TT],
				check_elem_in_clauses(Cl, El, T, TT);
			check_elem_in_clauses(Cl, El, T, L))).


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
