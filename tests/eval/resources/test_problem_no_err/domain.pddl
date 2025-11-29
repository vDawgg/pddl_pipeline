(define (domain blocks)

  (:requirements :strips :typing)

  (:types block)

  (:constants
   ; List all blocks here if known; otherwise use variables.
   ; Since no specific blocks are given, we'll leave this empty for now.
   )

  (:predicates
   (on ?b1 ?b2)           ; Block ?b1 is on top of block ?b2
   (clear ?b)             ; Block ?b has no blocks on top of it
   (holding ?b)           ; Robot is holding block ?b
   (at-table ?b)          ; Block ?b is on the table
   )

  (:action pick-up
    :parameters (?b - block)
    :precondition (and (at-table ?b) (clear ?b))
    :effect (and
      (not (at-table ?b))
      (holding ?b)
      (not (clear ?b))
      )
    )

  (:action put-down
    :parameters (?b - block)
    :precondition (holding ?b)
    :effect (and
      (at-table ?b)
      (not (holding ?b))
      (clear ?b)
      )
    )

  (:action stack
    :parameters (?b1 - block ?b2 - block)
    :precondition (and (holding ?b1) (clear ?b2))
    :effect (and
      (on ?b1 ?b2)
      (not (clear ?b2))
      (not (holding ?b1))
      )
    )

  (:action unstack
    :parameters (?b1 - block ?b2 - block)
    :precondition (and (on ?b1 ?b2) (clear ?b1))
    :effect (and
      (not (on ?b1 ?b2))
      (holding ?b1)
      (not (clear ?b1))
      )
    )
)
