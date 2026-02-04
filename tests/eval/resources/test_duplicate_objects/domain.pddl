(define (domain ring-sorting)
  (:requirements :typing :negative-preconditions :disjunctive-preconditions :action-costs)
  (:types ring peg robot-position)
  (:constants
    robot-arm
    peg1 peg2 peg3 peg4 peg5
    ring-red ring-blue ring-green ring-yellow ring-purple
  )
  (:predicates
    (at ?robot - robot-position ?pos - peg)
    (on ?ring - ring ?peg - peg)
    (gripper-open ?robot - robot-position)
    (gripper-closed ?robot - robot-position)
    (at-peg ?peg - peg)
    (color ?ring - ring ?c - color)
    (peg-color ?peg - peg ?c - color)
  )
  (:action move
    :parameters (?robot - robot-position ?goal - peg ?old-peg - peg)
    :precondition (and
      (at robot-arm ?old-peg)
      (at-peg ?goal)
      (not (at robot-arm ?goal))
    )
    :effect (and
      (at robot-arm ?goal)
      (not (at robot-arm ?old-peg))
    )
  )
  (:action pick
    :parameters (?robot - robot-position ?ring - ring ?peg - peg)
    :precondition (and
      (at robot-arm ?peg)
      (on ?ring ?peg)
      (gripper-open ?robot)
    )
    :effect (and
      (gripper-closed ?robot)
      (not (on ?ring ?peg))
    )
  )
  (:action place
    :parameters (?robot - robot-position ?ring - ring ?peg - peg)
    :precondition (and
      (at robot-arm ?peg)
      (gripper-closed ?robot)
    )
    :effect (and
      (gripper-open ?robot)
      (on ?ring ?peg)
    )
  )
)
