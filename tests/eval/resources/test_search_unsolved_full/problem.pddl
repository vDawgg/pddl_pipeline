(define (problem block-stacking-problem)

  (:domain block-stacking)

  (:objects
    A B C D E - block
  )

  (:init
    (at-table A)
    (on B A)
    (at-table C)
    (at-table D)
    (at-table E)
    (clear A)
    (clear C)
    (clear D)
    (clear E)
    (block A)
    (block B)
    (block C)
    (block D)
    (block E)
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
