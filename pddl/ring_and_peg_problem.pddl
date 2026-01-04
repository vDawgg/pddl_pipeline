(define (problem RING_AND_PEG_01)
    (:domain RING_AND_PEG)
    (:objects 
        red_ring green_ring blue_ring - ring
        ; TODO: The dummy pegs should also have colors associated with them
        ; NOTE: We will also have to make sure that the model can stick to the same object names so we can parse properly
        red_peg green_peg blue_peg dummy_1_peg dummy_2_peg start - peg
    )
    (:init 
        (onpeg red_ring dummy_1_peg) 
        (onpeg green_ring dummy_2_peg)
        (onpeg blue_ring red_peg)
        (pegempty blue_peg)
        (pegempty green_peg)
        (at nowhere)
        (handempty)
    )
    (:goal (and 
        (onpeg red_ring red_peg)
        (onpeg green_ring green_peg)
        (onpeg blue_ring blue_peg)
    ))
)
