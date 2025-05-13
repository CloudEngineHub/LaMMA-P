# Task Description: Put an Egg in the Fridge, and place a pot containing Apple slices into the refrigerator.

# GENERAL TASK DECOMPOSITION 
# Decompose and parallel subtasks where ever possible
# Independent subtasks:
# SubTask 1: Put an Egg in the Fridge. (Skills Required: GoToObject, PickupObject, OpenObject, PutObject, CloseObject)
# SubTask 2: Prepare Apple Slices. (Skills Required: GoToObject, PickupObject, SliceObject, PutObject)
# SubTask 3: Place the Pot with Apple Slices in the Fridge. (Skills Required: GoToObject, PickupObject, PutObject, OpenObject, CloseObject)
# We can parallelize SubTask 1 and SubTask 2 because they don't depend on each other.

#  action description from domain for tasks required
#subtask 1 Put an Egg in the Fridge
#GotoObject intakes robot as robot and object as object, 2 parameters. needs robot not inaction to perform the action. Action results in robot not inaction and robot at object location
#PickupObject intakes robot as robot, object as object, and location as object, 3 parameters. needs robot at the location of object, object at-location of object, and robot not inaction to perform the action. Action results in robot not inaction and robot holding object.
#OpenObject intakes robot as robot, object as object, 2 parameters. needs robot not inaction and robot at object to perform the action. Action results in robot not inaction and robot object-open the object.
#PutObject intakes robot as robot, object as object, and location as object, 3 parameters. needs robot holding the object, robot at destination location, and not inaction to perform the action. Action results in object at destination location, robot not holding object, and robot not inaction.
#CloseObject intaks robot as robot, object as object, 2 parameters. needs robot not inaction and robot at object to perform the action. Action results in robot not inaction and robot object-close the object.

#SubTask 2: Prepare Apple Slices.
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



