(define (problem blocks-reassembly)
  (:domain blocks)

  (:objects
    A B C D E - block
  )

  (:start
    (on B A)
    (at-table A)
    (at-table B)
    (at-table C)
    (at-table D)
    (at-table E)
    (clear A)
    (clear B)
    (clear C)
    (clear D)
    (clear E)
    (robot-free)
  )

  (:goal
    (and
      (on C B)
      (on D C)
      (on E D)
      (on A E)
      (not (on B A))
      (not (at-table B))
      (not (at-table A))
      (not (at-table C))
      (not (at-table D))
      (not (at-table E))
    )
  )
)
