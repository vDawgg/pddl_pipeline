(define (domain RING_AND_PEG)
    (:requirements :strips :typing :negative-preconditions)
    (:types ring peg)
    (:predicates 
        (at ?x - peg)
        (onpeg ?x - ring ?y - peg)
        (pegempty ?x - peg)
        (handempty)
        (holding ?x - ring)
    )

    (:action move
        ; NOTE: We need to model this in some way in the sim as well.
        ;       -> We could just ommit the from param when parsing as we do not need it for our
        ;          current move implementation.
        :parameters (?from - peg ?to - peg)
        :precondition (and (at ?from) (not (at ?to)))
        :effect (and 
            (not (at ?from))
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
