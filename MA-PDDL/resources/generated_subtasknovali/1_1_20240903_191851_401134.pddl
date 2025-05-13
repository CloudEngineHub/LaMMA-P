(define (problem put-apple-in-fridge)
  (:domain robot1)
  (:objects
    Robot1 - robot
    Apple Fridge - object
  )
  (:init
    (at Robot1 Robot1)
    (at-location Apple CounterTop)
    (object-close Robot1 Fridge)
    (not (inaction Robot1))
  )
  (:goal
    (and
      (at-location Apple Fridge)
      (object-close Robot1 Fridge)
      (not (holding Robot1 Apple))
    )
  )
)