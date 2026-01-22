(:problem sort-rings
  (:domain sort-rings)
  (:objects
    red_peg green_peg blue_peg pink_peg yellow_peg - peg
    red_ring green_ring blue_ring - ring
    robot - robot
  )
  (:init
    (at_robot default_position)
    (on red_ring pink_peg)
    (on green_ring yellow_peg)
    (on blue_ring red_peg)
    (not (holding red_ring))
    (not (holding green_ring))
    (not (holding blue_ring))
  )
  (:goal
    (and
      (on red_ring red_peg)
      (on green_ring green_peg)
      (on blue_ring blue_peg)
    )
  )
)
