(define (problem put_lettuce_in_fridge)
  (:domain robot2)
  (:objects
    Robot2 - robot
    Lettuce - object
    Fridge - object
    Location1 - object ; Assuming a location where the lettuce is initially located
    InsideFridge - object ; Assuming a location inside the fridge
  )
  
  (:init
    (at Robot2 Location1) ; Initial position of the robot is at Location1 where lettuce is located
    (at-location Lettuce Location1) ; Initial position of lettuce is at Location1
    (not (inaction Robot2)) ; Robot is not inaction initially
  )
  
  (:goal 
    (and 
      (at-location Lettuce InsideFridge) ; Goal is to have lettuce inside the fridge
      (object-close Robot2 Fridge) ; Ensure fridge is closed after putting lettuce inside it
    )
  )
)