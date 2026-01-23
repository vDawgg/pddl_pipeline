(define (problem blocks-reassembly)

  (:domain blocks)

  (:objects
    A B C D E - block
  )

  (:init
    (at-table A)
    (on B A)
    (at-table C)
    (at-table D)
    (at-table E)
    (block A)
    (block B)
    (block C)
    (block D)
    (block E)
    (clear A)
    (clear C)
    (clear D)
    (clear E)
    (clear B)
  )

  (:goal
    (and
      (on C B)
      (on D C)
      (on E D)
      (on A E)
      (not (on B A))
      (not (at-table B))
      )
  )

)
