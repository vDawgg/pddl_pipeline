(define (domain ring-sorting)

  (:requirements :strips :typing)

  (:types ring peg)

  (:predicates
    (at-robot-arm ?position)
    (on-ring ?ring ?peg)
    (holding-ring ?ring)
  )

  (:action move
    :parameters (?goal - peg)
    :precondition (and (not (at-robot-arm ?goal)) (exists (?peg - peg) (at-robot-arm ?peg)))
    :effect (and (not (at-robot-arm ?peg)) (at-robot-arm ?goal))
  )

  (:action pick
    :parameters (?ring - ring ?peg - peg)
    :precondition (and (at-robot-arm ?peg) (on-ring ?ring ?peg) (not (holding-ring ?ring)))
    :effect (and (not (on-ring ?ring ?peg)) (holding-ring ?ring))
  )

  (:action place
    :parameters (?ring - ring ?peg - peg)
    :precondition (and (at-robot-arm ?peg) (not (on-ring ?ring ?peg)) (holding-ring ?ring))
    :effect (and (not (holding-ring ?ring)) (on-ring ?ring ?peg))
  )

)
