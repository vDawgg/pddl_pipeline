(define (problem RING_AND_PEG_01)
    (:domain RING_AND_PEG)
    (:objects
        red_ring green_ring blue_ring
        red_peg green_peg blue_peg pink_peg yellow_peg start_position
    )
    (:init
        (ring red_ring) (ring green_ring) (ring blue_ring)
        (peg red_peg) (peg green_peg) (peg blue_peg) (peg pink_peg) (peg yellow_peg) (peg start_position)
        (onpeg red_ring pink_peg) (onpeg green_ring yellow_peg) (onpeg blue_ring red_peg)
        (pegempty blue_peg) (pegempty green_peg)
        (at start_position)
        (handempty)
    )
    (:goal (and (onpeg red_ring red_peg) (onpeg green_ring green_peg) (onpeg blue_ring blue_peg)))
)
