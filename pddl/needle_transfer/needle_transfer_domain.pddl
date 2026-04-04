(define (domain NEEDLE_TRANSFER)
  (:requirements :strips :disjunctive-preconditions :conditional-effects :negative-preconditions)

  (:predicates
    (robot ?r)
    (needle ?n)
    (position ?p)
    (ring ?r)
    (goal-position ?p)
    (initial-picker ?r)
    (last-ring ?r)
    (threading-complete)
    (robot-at ?robot ?position)
    (holding ?robot ?needle)
    (handover-pending ?robot ?needle)
    (needle-at ?needle ?position)
    (current-ring ?ring)
    (sequence ?ring_1 ?ring_2)
  )

  (:action move
    :parameters (?robot ?from ?to)
    :precondition (and
        (robot ?robot)
        (or (needle ?to) (ring ?to) (position ?to))
        (or
          (not (goal-position ?to))
          (threading-complete)
        )
        (robot-at ?robot ?from)
    )
    :effect (and
        (robot-at ?robot ?to)
        (not (robot-at ?robot ?from))
    )
  )

  (:action pick
    :parameters (?robot ?other_robot ?needle ?current_ring ?next_ring)
    :precondition (and
        (robot ?robot)
        (robot ?other_robot)
        (needle ?needle)
        (ring ?current_ring)
        (ring ?next_ring)
        (not (holding ?robot ?needle))
        (or
          ;; Initial pickup: the needle is on the ground and the robot moves to it.
          (and
            (initial-picker ?robot)
            (robot-at ?robot ?needle)
            (needle-at ?needle ?needle)
          )
          ;; Handoff pickup: both robots are at the current ring.
          (and
            (current-ring ?current_ring)
            (sequence ?current_ring ?next_ring)
            (robot-at ?robot ?current_ring)
            (robot-at ?other_robot ?current_ring)
            (holding ?other_robot ?needle)
          )
        )
    )
    :effect (and
        (holding ?robot ?needle)
        ;; Initial pickup: needle is removed from starting location
        (when (and
            (robot-at ?robot ?needle)
            (needle-at ?needle ?needle)
          )
          (not (needle-at ?needle ?needle))
        )
        ;; Handover has started, both arms are now holding the needle
        (when (and
            (current-ring ?current_ring)
            (sequence ?current_ring ?next_ring)
            (robot-at ?robot ?current_ring)
            (robot-at ?other_robot ?current_ring)
            (holding ?other_robot ?needle)
          )
          (handover-pending ?other_robot ?needle)
        )
    )
  )

  (:action place
    :parameters (?robot ?other_robot ?needle ?location ?current_ring ?next_ring)
    :precondition (and
        (robot ?robot)
        (robot ?other_robot)
        (needle ?needle)
        (holding ?robot ?needle)
        (robot-at ?robot ?location)
        (or
          ;; Ring transfer: receiving arm already grabbed the needle at the current ring
          (and
            (current-ring ?current_ring)
            (sequence ?current_ring ?next_ring)
            (ring ?location)
            (handover-pending ?robot ?needle)
            (robot-at ?other_robot ?current_ring)
            (holding ?other_robot ?needle)
          )
          ;; Robot at goal position. Needle can be dropped of
          (and
            (goal-position ?location)
            (threading-complete)
            (not (holding ?other_robot ?needle))
          )
        )
    )
    :effect (and
        (not (holding ?robot ?needle))
        (needle-at ?needle ?location)
        ;; Handover completed. Robot can move on to next ring in sequence or goal position
        (when (and
            (ring ?location)
            (current-ring ?current_ring)
            (sequence ?current_ring ?next_ring)
            (handover-pending ?robot ?needle)
          )
          (and
            (not (current-ring ?current_ring))
            (current-ring ?next_ring)
            (not (handover-pending ?robot ?needle))
          )
        )
        ;; Robot at goal position. Needle is placed and task complete
        (when (and
            (ring ?location)
            (current-ring ?current_ring)
            (last-ring ?current_ring)
          )
          (threading-complete)
        )
    )
  )
)
