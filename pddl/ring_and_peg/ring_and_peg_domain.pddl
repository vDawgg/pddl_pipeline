(define (domain RING_AND_PEG)
    (:requirements :strips)
    (:predicates
        (at ?x)
        (onpeg ?x ?y) (pegempty ?x)
        (handempty) (holding ?x)
        (peg ?x) (ring ?x)
    )

    (:action move
        :parameters (?from ?to)
        :precondition (and
            (at ?from)
            (peg ?from) (peg ?to)
        )
        :effect (and (not (at ?from)) (at ?to))
    )

    (:action pick
        :parameters (?x ?y)
        :precondition (and
            (at ?y)
            (onpeg ?x ?y)
            (handempty)
            (peg ?y) (ring ?x)
        )
        :effect(and
            (not (onpeg ?x ?y)) (pegempty ?y)
            (not (handempty)) (holding ?x)
        )
    )

    (:action place
        :parameters (?x ?y)
        :precondition (and
            (at ?y)
            (holding ?x)
            (pegempty ?y)
            (peg ?y)
            (ring ?x)
        )
        :effect (and
            (not (holding ?x)) (handempty)
            (onpeg ?x ?y) (not (pegempty ?y))
        )
    )
)
