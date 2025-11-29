(define (problem block-stack-reassembly)

(:domain block-stack)

(:objects
    A B C D E - block
)

(:init
    (on-table A)
    (on B A)
    (on-table C)
    (on-table D)
    (on-table E)
    (free-arm)
)

(:goal
    (and
        (on-table B)
        (on C B)
        (on D C)
        (on E D)
        (on A E)
    )
)

)
