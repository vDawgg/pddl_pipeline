(define (domain new-blocks)

  (:requirements :strips :typing)

  (:types block)

  (:predicates
   (on ?b1 ?b2)
   (clear ?b)
   (holding ?b)
   (at-table ?b)
   )

  (:action pick-up
    :parameters (?b - block)
    :precondition (and (at-table ?b) (clear ?b))
    :effect (and
      (not (at-table ?b))
      (holding ?b)
      )
    )

)
