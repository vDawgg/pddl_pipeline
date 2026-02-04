0:	(define (domain blocks)
1:	
2:	  (:requirements :strips :typing)
3:	
4:	  (:types block)
5:	
6:	  (:predicates
7:	   (on ?b1 ?b2)
8:	   (clear ?b)
9:	   (holding ?b)
10:	   (at-table ?b)
11:	   )
12:	
13:	  (:action pick-up
14:	    :parameters (?b - block)
15:	    :precondition (and (at-table ?b) (clear ?b))
16:	    :effect (and
17:	      (not (at-table ?b))
18:	      (holding ?b)
19:	      )
20:	    )
21:	
22:	)
