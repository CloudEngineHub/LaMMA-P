# Task Description: Put an Egg in the Fridge, and place a pot containing Apple slices into the refrigerator.

# GENERAL TASK DECOMPOSITION 
# Decompose and parallel subtasks where ever possible
# Independent subtasks:
# SubTask 1: Put an Egg in the Fridge. (Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject)
# SubTask 2: Prepare Apple Slices. (Skills Required: GoToObject, PickupObject, SliceObject, PutObject)
# SubTask 3: Place the Pot with Apple Slices in the Fridge. (Skills Required: GoToObject, PickupObject, PutObject, OpenObject, CloseObject)
# We can parallelize SubTask 1 and SubTask 2 because they don't depend on each other.

# problem pddl file  
#subtask 1 Put an Egg in the Fridge
(define (problem put-egg-in-fridge)
  (:domain allactionrobot)
   
  (:objects
    allactionrobot - robot
    egg - object
    fridge - object
    kitchen -object
    table -object

  )
  
  (:init
    (not(inaction allactionrobot))
    (at-location egg table)
    (at-location fridge kitchen)
    (not (holding allactionrobot egg))
    (not (object-open allactionrobot fridge))
  )
  
  (:goal
    (and
      (at-location fridge egg)
      (not (holding allactionrobot egg))
    )
  )
)

# subtask 2: Prepare Apple Slices 
(define (problem prepare-apple-slices)
  (:domain allactionrobot)
  
  (:objects
    allactionrobot - robot
    apple - object
    knife - object
    counter - object
    plate - object
  )
  
  (:init
    (not(inaction allactionrobot))
    (at-location apple table)
    (at-location knife kitchen)
    (at-location counter kitchen)
    (at-location plate kitchen)
    (not (holding allactionrobot apple))
    (not (holding allactionrobot knife))
    (not (object-sliced apple))
    (not (on apple plate))
  )
  
  (:goal
    (and
      (not(inaction allactionrobot))
      (object-sliced apple)
      (on apple plate)
      (not (holding allactionrobot apple))
      (not (holding allactionrobot knife))
    )
  )
)

#subtask 3: Place the Pot with Apple Slices in the Fridge
(define (problem place-pot-in-fridge)
  (:domain allactionrobot)
  
  (:objects
    allactionrobot - robot
    pot - object
    fridge - object
    counter - object
  )
  
  (:init
    (not(inaction allactionrobot))
    (at pot counter)
    (at fridge kitchen)
    (not (holding allactionrobot pot))
    (object-closed fridge)
  )
  
  (:goal
    (and
      (not(inaction allactionrobot))
      (at pot fridge)
      (not (holding allactionrobot pot))
      (object-closed fridge)
    )
  )
)



# Task Put an Egg in the Fridge, and place a pot containing Apple slices into the refrigerator is done
#IMPORTANT strictly follow the structure, stop generate after Task is done.







# Task Description: Make a sandwich with sliced lettuce, sliced tomato, sliced bread and serve it on a washed plate.

# GENERAL TASK DECOMPOSITION
# Decompose and parallelize subtasks where ever possible
# Independent subtasks:
# SubTask 1: Slice the Lettuce, Tomato, and Bread. (Skills Required: GoToObject, PickupObject, SliceObject, PutObject)
# SubTask 2: Wash the Plate. (Skills Required: GoToObject, PickupObject, PutObject, SwitchOn, SwitchOff)
# SubTask 3: Assemble the Sandwich. (Skills Required: GoToObject, PickupObject, PutObject)
# We can parallelize SubTask 1 and SubTask 2 because they don't depend on each other.

# problem pddl file
#problem file for subtask 1 Slice the Lettuce, Tomato, and Bread
(define (problem slice-ingredients)
  (:domain allactionrobot)
  
  (:objects
    allactionrobot - robot
    lettuce - object
    tomato - object
    bread - object
    knife - object
    counter - object
  )
  
  (:init
    (not (inaction allactionrobot))
    (at-location lettuce counter)
    (at-location tomato counter)
    (at-location bread counter)
    (at-location knife kitchen)
    (not (holding allactionrobot lettuce))
    (not (holding allactionrobot tomato))
    (not (holding allactionrobot bread))
    (not (holding allactionrobot knife))

  )
  
  (:goal
    (and
      (not (inaction allactionrobot))
      (sliced lettuce)
      (sliced tomato)
      (sliced bread)    
      (at-location lettuce counter)
      (at-location tomato counter)
      (at-location bread counter)
      (at-location knife kitchen)
      (not (holding allactionrobot lettuce))
      (not (holding allactionrobot tomato))
      (not (holding allactionrobot bread))
      (not (holding allactionrobot knife))

    )
  )
)
# problem file for subtask2  Wash the Plate.
(define (problem wash-plate)
  (:domain allactionrobot)
  
  (:objects
    allactionrobot - robot
    plate  - object
    sink - object
    faucet - object
    drying-rack - object
    table -object
    kitchen -object

  )
  
  (:init
    (not (inaction allactionrobot))
    (at-location plate table)
    (at-location sink kitchen)
    (at-location faucet kitchen)
    (at-location drying-rack kitchen)
    (not (holding allactionrobot plate))
    (not (cleaned plate))
  )
  
  (:goal
    (and
      (cleaned plate)
      (at-location plate drying-rack)
      (not (holding allactionrobot plate))
      (switch-off faucet)
    )
  )
)
(define (problem assemble-sandwich)
  (:domain allactionrobot)
  
  (:objects
    allactionrobot - robot
    bread - object
    lettuce - object
    tomato - object
    cheese - object
    sandwich - object
    counter - object
  )
  
  (:init
    (not (inaction allactionrobot))
    (at bread counter)
    (at lettuce counter)
    (at tomato counter)
    (at cheese counter)
    (at sandwich counter)
    (not (holding allactionrobot bread))
    (not (holding allactionrobot lettuce))
    (not (holding allactionrobot tomato))
    (not (holding allactionrobot cheese))
    
  )
  
  (:goal
    (and
      (not (inaction allactionrobot))
      (not (holding allactionrobot bread))
      (not (holding allactionrobot lettuce))
      (not (holding allactionrobot tomato))
      (not (holding allactionrobot cheese))

    )
  )
)

# Task Make a sandwich with sliced lettuce, sliced tomato, sliced bread and serve it on a washed plate is done
#IMPORTANT strictly follow the structure, stop generate after Task is done.




