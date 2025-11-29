(define (domain block-stack)

(:requirements :typing :negative-preconditions :disjunctive-preconditions)

(:types
    block
)

(:constants
    ; List all blocks here
)

(:predicates
    ; Block x is on the table
    (on-table ?x - block)
    ; Block x is on top of block y
    (on ?x ?y - block)
    ; Block x is clear (nothing is on top of it)
    (clear ?x - block)
    ; Block x is holding by the robot arm
    (holding ?x - block)
    ; Block x is gripped by the robot arm (not necessarily holding it)
    (gripped ?x - block)
    ; Block x is attached to a stack (i.e., part of a tower)
    (attached ?x - block)
    ; Robot arm is free (not gripping anything)
    (free-arm)
)

(:actions
    ; Pick up block x from the table or from top of another block
    (pick-up
        :parameters (?x - block)
        :precondition (and (on-table ?x) (clear ?x) (free-arm))
        :effect (and
            (not (on-table ?x))
            (holding ?x)
            (not (clear ?x))
            (not (free-arm))
            (gripped ?x)
        )
    )

    ; Put down block x on the table or on top of another block
    (put-down
        :parameters (?x - block)
        :precondition (and (holding ?x) (clear ?y) (not (attached ?x)))
        :effect (and
            (not (holding ?x))
            (on-table ?x)
            (clear ?y)
            (not (gripped ?x))
            (free-arm)
        )
    )

    ; Stack block x on top of block y
    (stack
        :parameters (?x - block ?y - block)
        :precondition (and (holding ?x) (on ?y ?z) (clear ?y) (not (attached ?x)))
        :effect (and
            (not (holding ?x))
            (on ?x ?y)
            (clear ?z)
            (not (gripped ?x))
            (free-arm)
            (attached ?x)
        )
    )

    ; Unstack block x from block y
    (unstack
        :parameters (?x - block ?y - block)
        :precondition (and (on ?x ?y) (clear ?x) (not (attached ?x)))
        :effect (and
            (not (on ?x ?y))
            (on-table ?x)
            (clear ?y)
            (not (attached ?x))
            (gripped ?x)
            (free-arm)
        )
    )
)

)
