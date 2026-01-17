(define (domain ring-sorting)

 (:requirements :strips :typing)

 (:types ring peg - object
         red green blue pink yellow - color)

 (:predicates (at ?r - ring ?p - peg)
             (free ?p - peg)
             (gripper-closed)
             (gripper-open))

 (:action move
  :parameters (?goal - peg)
  :precondition (and (not (at ?robot arm ?goal)) (free ?goal))
  :effect (at ?robot arm ?goal))

 (:action pick
  :parameters (?r - ring ?p - peg)
  :precondition (and (gripper-open) (at ?r ?p))
  :effect (not (gripper-open)))

 (:action place
  :parameters (?r - ring ?p - peg)
  :precondition (and (gripper-closed) (free ?p))
  :effect (and (gripper-open) (at ?r ?p)))
