# Visprog + GPT DSL results

(visprog) sanideshmukh@Sanis-MacBook-Pro-242 visprog_new % python compare_images_dsl.py 
Registering VQA step
Registering EVAL step
Registering RESULT step
Registering FIND step
Registering COUNT step
Registering FILTER step
Registering EXISTS step

🧾 Raw GPT Response:
```json
[
  "How many boats are visible in the water?",
  "What is the color of the ferry?",
  "What is the shape of the sails on the building?",
  "Is there a path visible near the water?",
  "What is the material of the large building's exterior?",
  "How many green areas are visible in the scene?",
  "Are there any people visible on the ferry?",
  "What is the size of the white sailboat in the water?"
]
```
Registering VQA step
Registering EVAL step
Registering RESULT step
Registering FIND step
Registering COUNT step
Registering FILTER step
Registering EXISTS step

🔎 Executing Symbolic Programs via VisProg
============================================================

→ Question 1: How many boats are visible in the water?
[LEFT DSL]
boats = FIND(image=LEFT, object="boats")
num_boats = COUNT(region=boats)
result = RESULT(var=num_boats)
FIND
COUNT
RESULT

[RIGHT DSL]
boats = FIND(image=RIGHT, object="boats")
num_boats = COUNT(region=boats)
result = RESULT(var=num_boats)
FIND
COUNT
RESULT

LEFT : 8
RIGHT: 13
➤ Different? → Yes

→ Question 2: What is the color of the ferry?
[LEFT DSL]
ferry_region = FIND(image=LEFT, object="ferry")
ferry_color = VQA(image=ferry_region, question="What is the color of this?")
result = RESULT(var=ferry_color)
FIND
VQA
/Users/sanideshmukh/miniforge3/envs/visprog/lib/python3.10/site-packages/transformers/generation/utils.py:1288: UserWarning: Using `max_length`'s default (20) to control the generation length. This behaviour is deprecated and will be removed from the config in v5 of Transformers -- we recommend using `max_new_tokens` to control the maximum length of the generation.
  warnings.warn(
RESULT

[RIGHT DSL]
ferry = FIND(image=RIGHT, object="ferry")
color = VQA(image=RIGHT, question="What is the color of the ferry?")
result = RESULT(var=color)
FIND
VQA
RESULT

LEFT : blue
RIGHT: white
➤ Different? → Yes

→ Question 3: What is the shape of the sails on the building?
[LEFT DSL]
sails_shape = VQA(image=LEFT, question="What is the shape of the sails on the building?")
result = RESULT(var=sails_shape)
VQA
RESULT

[RIGHT DSL]
sails_region = FIND(image=RIGHT, object="sails")
sails_shape = VQA(image=sails_region, question="What is the shape of this object?")
result = RESULT(var=sails_shape)
FIND
VQA
RESULT

LEFT : curved
RIGHT: round
➤ Different? → Yes

→ Question 4: Is there a path visible near the water?
[LEFT DSL]
path_region = FIND(image=LEFT, object="path")
water_region = FIND(image=LEFT, object="water")
path_near_water = EXISTS(region=path_region & water_region)
result = RESULT(var=path_near_water)
FIND
FIND
EXISTS
Error: 'path_region & water_region'

→ Question 5: What is the material of the large building's exterior?
[LEFT DSL]
output_var = VQA(image=LEFT, question="What is the material of the large building's exterior?")
result = RESULT(var=output_var)
VQA
RESULT

[RIGHT DSL]
output_var = VQA(image=RIGHT, question="What is the material of the large building's exterior?")
result = RESULT(var=output_var)
VQA
RESULT

LEFT : metal
RIGHT: metal
➤ Different? → No

→ Question 6: How many green areas are visible in the scene?
[LEFT DSL]
green_areas = FIND(image=LEFT, object="green areas")
count_green_areas = COUNT(region=green_areas)
result = RESULT(var=count_green_areas)
FIND
COUNT
RESULT

[RIGHT DSL]
green_areas = FIND(image=RIGHT, object="green areas")
green_areas_count = COUNT(region=green_areas)
result = RESULT(var=green_areas_count)
FIND
COUNT
RESULT

LEFT : 3
RIGHT: 4
➤ Different? → Yes

→ Question 7: Are there any people visible on the ferry?
[LEFT DSL]
people_region = FIND(image=LEFT, object="people")
people_exist = EXISTS(region=people_region)
result = RESULT(var=people_exist)
FIND
EXISTS
RESULT

[RIGHT DSL]
people_region = FIND(image=RIGHT, object="people")
people_exist = EXISTS(region=people_region)
result = RESULT(var=people_exist)
FIND
EXISTS
RESULT

LEFT : True
RIGHT: True
➤ Different? → No

→ Question 8: What is the size of the white sailboat in the water?
[LEFT DSL]
sailboat_region = FIND(image=LEFT, object="white sailboat")
sailboat_size = VQA(image=sailboat_region, question="What is the size of this object?")
result = RESULT(var=sailboat_size)
FIND
VQA
RESULT

[RIGHT DSL]
sailboat_region = FIND(image=RIGHT, object="white sailboat")
sailboat_size = VQA(image=sailboat_region, question="What is the size of this?")
result = RESULT(var=sailboat_size)
FIND
VQA
RESULT

LEFT : small
RIGHT: small
➤ Different? → No

TOTAL DIFFERENCES FOUND: 4
(visprog) sanideshmukh@Sanis-MacBook-Pro-242 visprog_new % 

# GPT ONLY

visprog) sanideshmukh@Sanis-MacBook-Pro-242 visprog_new % python compare_images_chat.py             
📡 Getting localized comparison questions from GPT-4o...

🧾 Raw GPT Response:
```json
[
    "Is there a boat near the building?",
    "How many sailboats can be seen in the water?",
    "What is the primary color of the large building?",
    "Are the sails of the building visible?",
    "Is the boat moving upward or downward?",
    "Is there a bridge in the background?",
    "How many white structures are on the water?",
    "Is the opera house upright or tilted?",
    "Is there a green area in the middle ground?",
    "Is there a person visible on the shore?",
    "Is the sky clear or cloudy?",
    "Is there a tower in the distance?",
    "What color is the water closest to the building?",
    "Are there trees lining the walkway?",
    "Is the coastline curved or straight?"
]
```

📝 Questions to test:
  1. Is there a boat near the building?
  2. How many sailboats can be seen in the water?
  3. What is the primary color of the large building?
  4. Are the sails of the building visible?
  5. Is the boat moving upward or downward?
  6. Is there a bridge in the background?
  7. How many white structures are on the water?
  8. Is the opera house upright or tilted?
  9. Is there a green area in the middle ground?
  10. Is there a person visible on the shore?
  11. Is the sky clear or cloudy?
  12. Is there a tower in the distance?
  13. What color is the water closest to the building?
  14. Are there trees lining the walkway?
  15. Is the coastline curved or straight?

🧠 GPT-Generated Questions & GPT-4o VQA Results
============================================================

 ->> Question 1: Is there a boat near the building?
  LEFT : Yes.
  RIGHT: Yes.
  ➤ Different? → No

 ->> Question 2: How many sailboats can be seen in the water?
  LEFT : One.
  RIGHT: Four.
  ➤ Different? → Yes

 ->> Question 3: What is the primary color of the large building?
  LEFT : White
  RIGHT: White
  ➤ Different? → No

 ->> Question 4: Are the sails of the building visible?
  LEFT : Yes.
  RIGHT: Yes.
  ➤ Different? → No

 ->> Question 5: Is the boat moving upward or downward?
  LEFT : Downward.
  RIGHT: Upward.
  ➤ Different? → Yes

 ->> Question 6: Is there a bridge in the background?
  LEFT : Yes.
  RIGHT: Yes.
  ➤ Different? → No

 ->> Question 7: How many white structures are on the water?
  LEFT : One.
  RIGHT: One.
  ➤ Different? → No

 ->> Question 8: Is the opera house upright or tilted?
  LEFT : Upright.
  RIGHT: Upright.
  ➤ Different? → No

 ->> Question 9: Is there a green area in the middle ground?
  LEFT : Yes.
  RIGHT: Yes.
  ➤ Different? → No

 ->> Question 10: Is there a person visible on the shore?
  LEFT : Yes.
  RIGHT: Yes.
  ➤ Different? → No

 ->> Question 11: Is the sky clear or cloudy?
  LEFT : Clear.
  RIGHT: Clear.
  ➤ Different? → No

 ->> Question 12: Is there a tower in the distance?
  LEFT : Yes.
  RIGHT: Yes.
  ➤ Different? → No

 ->> Question 13: What color is the water closest to the building?
  LEFT : Blue.
  RIGHT: Blue.
  ➤ Different? → No

 ->> Question 14: Are there trees lining the walkway?
  LEFT : No.
  RIGHT: Yes.
  ➤ Different? → Yes

 ->> Question 15: Is the coastline curved or straight?
  LEFT : Curved.
  RIGHT: Curved.
  ➤ Different? → No
TOTAL DIFFERENCES FOUND: 3
(visprog) sanideshmukh@Sanis-MacBook-Pro-242 visprog_new % c