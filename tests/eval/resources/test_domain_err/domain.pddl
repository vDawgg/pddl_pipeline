(define (domain block-world)

  (:requirements :strips :typing)

  (:types block)

  (:predicates
    (on ?x ?y)       ; ?x is on top of ?y
    (clear ?x)        ; ?x has no blocks on top of it
    (holding ?x)      ; robot is holding block ?x
    (at-table ?x)     ; block ?x is at the table (i.e., not stacked)
    (block ?x)        ; ?x is a block
  )

  (:actions
    ; Pick up block x if it is at the table and clear.
    (pick-up
      :parameters (?x - block)
      :precondition (and (at-table ?x) (clear ?x))
      :effect (and
        (not (at-table ?x))
        (not (clear ?x))
        (holding ?x)
      )
    )

    ; Put down block x if robot is holding it.
    (put-down
      :parameters (?x - block)
      :precondition (holding ?x)
      :effect (and
        (not (holding ?x))
        (at-table ?x)
        (clear ?x)
      )
    )

    ; Stack block x on top of block y if x is held and y is clear.
    (stack
      :parameters (?x - block ?y - block)
      :precondition (and (holding ?x) (clear ?y))
      :effect (and
        (not (holding ?x))
        (on ?x ?y)
        (not (clear ?y))
      )
    )

    ; Unstack block x from block y if x is on top of y and y is not clear.
    (unstack
      :parameters (?x - block ?y - block)
      :precondition (and (on ?x ?y) (not (clear ?y)))
      :effect (and
        (not (on ?x ?y))
        (at-table ?x)
        (clear ?x)
        (clear ?y)
      )
    )
  )

)
