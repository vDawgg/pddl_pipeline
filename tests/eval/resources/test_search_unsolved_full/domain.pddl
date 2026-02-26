(define (domain block-stacking)

  (:requirements :typing :negative-preconditions :disjunctive-preconditions)

  (:types block)

  (:predicates
    (on ?b1 ?b2)           ; block ?b1 is on top of block ?b2
    (clear ?b)             ; block ?b has no blocks on top of it
    (holding ?b)           ; robot is holding block ?b
    (at-table ?b)          ; block ?b is at the table (not stacked)
    (stacked ?b)           ; block ?b is part of a stack (not at table)
    (on-table ?b)          ; block ?b is directly on the table
    (block ?b)             ; block ?b is a block
  )

  (:action pick-up
    :parameters (?b - block)
    :precondition (and (at-table ?b) (clear ?b) (not (holding ?b)))
    :effect (and
      (not (at-table ?b))
      (holding ?b)
      (not (clear ?b))
    )
  )

  (:action put-down
    :parameters (?b - block)
    :precondition (and (holding ?b) (not (on-table ?b)))
    :effect (and
      (not (holding ?b))
      (at-table ?b)
      (clear ?b)
      (not (stacked ?b))
    )
  )

  (:action stack
    :parameters (?b1 - block ?b2 - block)
    :precondition (and (holding ?b1) (clear ?b2) (not (on ?b1 ?b2)))
    :effect (and
      (not (holding ?b1))
      (on ?b1 ?b2)
      (not (clear ?b2))
      (stacked ?b2)
      (not (at-table ?b1))
      (not (on-table ?b1))
    )
  )

  (:action unstack
    :parameters (?b1 - block ?b2 - block)
    :precondition (and (on ?b1 ?b2) (clear ?b1))
    :effect (and
      (not (on ?b1 ?b2))
      (holding ?b1)
      (not (clear ?b1))
      (at-table ?b1)
      (not (stacked ?b1))
      (not (on-table ?b1))
    )
  )

)
