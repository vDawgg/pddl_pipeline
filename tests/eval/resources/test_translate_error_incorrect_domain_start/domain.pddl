(:domain sort-rings
  (:requirements :typing)
  (:types ring peg)
  (:predicates
    (at_robot ?r - robot)
    (on ?r - ring ?p - peg)
    (holding ?r - ring)
    (at ?r - ring ?p - peg)
    (goal_achieved)
  )
  (:action move
    :parameters (?goal - peg)
    :precondition (at_robot ?goal)
    :effect (at_robot ?goal)
  )
  (:action pick
    :parameters (?r - ring)
    :precondition (on ?r ?p) (at_robot ?p)
    :effect (holding ?r) (not (on ?r ?p))
  )
  (:action place
    :parameters (?r - ring ?p - peg)
    :precondition (holding ?r) (at_robot ?p)
    :effect (on ?r ?p) (not (holding ?r))
  )
)
