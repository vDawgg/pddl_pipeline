(define (domain RING_AND_PEG)
    (:requirements :strips :typing :negative-preconditions :conditional-effects)
    (:types ring peg)
    (:predicates
        (at ?x - peg)
        (onpeg ?x - ring ?y - peg)
        (pegempty ?x - peg)
        (handempty)
        (holding ?x - ring)
    )

    (:action move
        :parameters (?to - peg)
        :precondition (and (not (at ?to)))
        :effect (and
            (forall (?p - peg) (not (at ?p)))
            (at ?to)
        )
    )

    (:action pick
        :parameters (?x - ring ?y - peg)
        :precondition (and (at ?y) (onpeg ?x ?y) (handempty))
        :effect(and
            (not (onpeg ?x ?y))
            (not (handempty))
            (holding ?x)
            (pegempty ?y)
        )
    )

    (:action place
        :parameters (?x - ring ?y - peg)
        :precondition (and (at ?y) (holding ?x) (pegempty ?y))
        :effect (and
            (not (holding ?x))
            (handempty)
            (onpeg ?x ?y)
            (not (pegempty ?y))
        )
    )
)
