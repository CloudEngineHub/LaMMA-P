import copy
import glob
import json
import os
import argparse
from pathlib import Path
from datetime import datetime
import random
import subprocess
import time
import re
import shutil
import sys

import openai
import ai2thor.controller

import sys
sys.path.append(".")

import resources.actions as actions
import resources.robots as robots


def LM(prompt, gpt_version, max_tokens=128, temperature=0, stop=None, logprobs=1, frequency_penalty=0):
    time.sleep(60)
    
    if "gpt" not in gpt_version:
        response = openai.Completion.create(model=gpt_version, 
                                            prompt=prompt, 
                                            max_tokens=max_tokens, 
                                            temperature=temperature, 
                                            stop=stop, 
                                            logprobs=logprobs, 
                                            frequency_penalty = frequency_penalty)
                                            
        
        return response, response["choices"][0]["text"].strip()
    
    else:
        response = openai.ChatCompletion.create(model=gpt_version, 
                                            messages=prompt, 
                                            max_tokens=max_tokens, 
                                            temperature=temperature, 
                                            frequency_penalty = frequency_penalty)                                      
        return response, response["choices"][0]["message"]["content"].strip()

def set_api_key(openai_api_key):
    openai.api_key = Path(openai_api_key + '.txt').read_text()

# Function returns object list with name and properties.
def convert_to_dict_objprop(objs, obj_mass):
    objs_dict = []
    for i, obj in enumerate(objs):
        obj_dict = {'name': obj , 'mass' : obj_mass[i]}
        # obj_dict = {'name': obj , 'mass' : 1.0}
        objs_dict.append(obj_dict)
    return objs_dict
""""
def get_ai2_thor_objects(floor_plan_id):
    # connector to ai2thor to get object list
    controller = ai2thor.controller.Controller(scene="FloorPlan"+str(floor_plan_id))
    obj = list([obj["objectType"] for obj in controller.last_event.metadata["objects"]])
    obj_mass = list([obj["mass"] for obj in controller.last_event.metadata["objects"]])
    controller.stop()
    obj = convert_to_dict_objprop(obj, obj_mass)
    return obj""""

def calculate_task_completion_rate():
    TC = 0
    total_subtasks = 0
    base_path = os.path.join(os.getcwd(), "resources", "generated_subtask")

    for file_path in glob.glob(os.path.join(base_path, '*_plan.txt')):
        total_subtasks += 1
        with open(file_path, 'r') as file:
            content = file.read()
            TC += content.count('Solution found!')
    
    return TC, total_subtasks

def split_pddl_tasks(code_plan):
    

    directory = os.path.join(os.getcwd(), "resources", "generated_subtask")
    os.makedirs(directory, exist_ok=True)

    for i, plan in enumerate(code_plan):
        tasks = re.split(r"\s*\(define\s*\(problem", plan)

        for j, task in enumerate(tasks[1:]):
            print("current task is:")
            print(task)
            task = "(define (problem" + task  # Add back the removed "(define "
            task = balance_parentheses(task)
            task_name_line = task.split("\n", 1)[0]  # Get the first line of the task
            print(task)

            # Generate the filename using index as the first letter
            match = re.search(r'\(problem\s+(\w+)\)', task, re.IGNORECASE)

            if match:
                task_name = match.group(1)
                filename = f"{i+1}_{j+1}_{task_name}.pddl"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{i+1}_{j+1}_{timestamp}.pddl"


            filepath = os.path.join(directory, filename)
            # Save the task to the file

        # Construct the filename using the timestamp

        
            with open(filepath, 'w') as file:
                file.write(task)




def balance_parentheses(content):
    open_count = 0
    problem_content = ""
    start_index = -1
    end_index = -1

    for i, char in enumerate(content):
        if char == '(':
            if open_count == 0:
                start_index = i
            open_count += 1
        elif char == ')':
            open_count -= 1
            if open_count == 0:
                end_index = i
                break
    
    if start_index != -1 and end_index != -1:
        problem_content = content[start_index:end_index+1]
    
    return problem_content


def read_file(file_path):
    
    with open(file_path, 'r') as file:
        return file.read()

def run_llmvalidator():
    base_path = os.path.join(os.getcwd(), "resources")
    

    problem_path = os.path.join(base_path, "generated_subtask")

    problem_files = [f for f in os.listdir(problem_path) if f.endswith('.pddl')]
    

    
    for problem_file in problem_files:
        problem_file_full = os.path.join(problem_path, problem_file)
        domain_name = extract_domain_name(problem_file_full)
        if not domain_name:
            print(f"No domain specified in {problem_file}")
            continue


        domain_file = find_domain_file(base_path, domain_name)
        if not domain_file:
            print(f"No domain file found for domain {domain_name} required by {problem_file}")
            continue

        domain_content = read_file(domain_file)
        problem_content = read_file(problem_file_full)

        prompt = (f"Domain Description:\n"
                  f"{domain_content}\n\n"
                  f"Problem Description:\n"
                  f"{problem_content}\n\n"
                  "Validate the preconditions in problem file to ensure all precondition listed object is included and also in domain file, and go over structure to check the parenthesis and syntext.  Check and return only the validated problem file."
        )

        code_plan =[]

        
        if "gpt" not in args.gpt_version:
            # older versions of GPT
            _, text = LM(prompt, args.gpt_version, max_tokens=1000, stop=["def"], frequency_penalty=0.30)
        else:            
            # using variants of gpt 4 or 3.5
            messages = [{"role": "system", "content": "You are a Robot PDDL problem Expert"},{"role": "user", "content": prompt}]
            _, text = LM(messages, args.gpt_version, max_tokens=1400, frequency_penalty=0.4)


        code_plan.append(text)
        split_pddl_tasks(code_plan)


        return None
##############fastdownward as default planner
def run_planners():
    
    base_path = os.path.join(os.getcwd(), "resources")
    planner_path = os.path.join(os.getcwd(), "downward", "fast-downward.py") # fast-downward
    
    problem_path = os.path.join(base_path, "generated_subtask")

    problem_files = [f for f in os.listdir(problem_path) if f.endswith('.pddl')]

    for idx, problem_file in enumerate(problem_files):
        problem_file_full = os.path.join(problem_path, problem_file)
        domain_name = extract_domain_name(problem_file_full)
        if not domain_name:
            print(f"No domain specified in {problem_file}")
            continue

        domain_file = find_domain_file(base_path, domain_name)
        if not domain_file:
            print(f"No domain file found for domain {domain_name} required by {problem_file}")
            continue

        command = [
            planner_path,
            "--alias",
            "seq-opt-lmcut",
            domain_file,
            problem_file_full,  # adjust as necessary

        ]
### issue std out overwrite


        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        #result2 = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        
        output_file_temp = os.path.join(problem_path, problem_file.replace('.pddl', '_plan.txt'))
        with open(output_file_temp, 'w') as f:
            f.write(result.stdout)

        ##with open(output_file_timestamped, 'w') as f:
            #f.write(result2.stdout)

        # Print any errors
        if result.stderr:
            print(f"Errors for {problem_file}:", result.stderr)


def split_and_store_tasks(content): 
    print("the plan for current run is\n")
    print(content)
    # Define the pattern for detecting problem content summary and sequence of operations
    summary_pattern = re.compile(r'#?\s*Problem content summary\s*:?(.*?)(?=#?\s*Sequence of Operations?\s*:?)', re.DOTALL | re.IGNORECASE)
    sequence_pattern = re.compile(r'#?\s*Sequence of Operations?\s*:\s*(.*)', re.DOTALL | re.IGNORECASE)

    # Extract the problem content summary
    summary_match = summary_pattern.search(content)
    if summary_match:
        problem_summary = summary_match.group(1).strip()
    else:
        problem_summary = "failed to extract1"

    # Extract the sequence of operations
    sequence_match = sequence_pattern.search(content)
    if sequence_match:
        sequence_operations = sequence_match.group(1).strip()
    else:
        sequence_operations = "failed to extract2"


    # Split the problem summary into individual subtasks
    subtasks = re.split(r'\s*\w?\s*(?:\n)?\s*(?=#\s*subtask\s*:?)\s*', problem_summary, flags=re.IGNORECASE)
    subtasks = [subtask.strip() for subtask in subtasks if subtask.strip()]


    return subtasks, sequence_operations

#def team_robotproblem(assigned_robots):


def problemextracting(subtasks):
    filenames = []
    for subtask in subtasks:

        print("subtasks are:")
        print(subtask)
        # Initialize robotassignnumber with a default value
        robotassignnumber = "unknown_robot.pddl"


        # Extract assigned robots using regex
        pattern = re.compile(r"\s*\*{0,2}\s*Assigned\s*Robot\s*\??:?\*{0,2}\s*\??(.*?)\s*\*{0,2}\s*Objects\s*Involved\s*:\??\*{0,2}", re.DOTALL | re.IGNORECASE)
        assigned_robots_match = pattern.search(subtask)
        print("assigned_robots_match is")
        print(assigned_robots_match)


        if assigned_robots_match:
            #modify to not force allactionrobot/ team robot as allactionrobot.
            assigned_robots = assigned_robots_match.group(1).strip()
            if "team" in assigned_robots.lower() or "allactionrobot" in assigned_robots.lower():
                print("this is a team task")
                #team_robotproblem(assigned_robots)
                # Handle team robots
                all_domain_contents = ""
                team_pattern = re.compile(r"\s*\*{0,2}\s*robot\s*\??\s*(\d+)\*{0,2}", re.IGNORECASE)
                robot_numbers = team_pattern.findall(assigned_robots)
                normalized_robot_numbers = [f"robot{num}" for num in robot_numbers]
                for robot in normalized_robot_numbers:
                    domain_path = os.path.join(os.getcwd(), "resources", f"{robot}.pddl")

                    with open(domain_path, 'r') as file:
                        domain_content = file.read()
                        file.close()
                        all_domain_contents += domain_content

                problem_fileexamplepath = os.getcwd() + "/data/pythonic_plans/" + args.prompt_allocation_set + "_teamproblem.py"
                problem_fileexample = open(problem_fileexamplepath, "r")
                problem_examplecontent = problem_fileexample.read()
                problem_fileexample.close()

                prompt = "\n"
                prompt += problem_examplecontent
                prompt = "Strictly follow the structure and finish the tasks like example"
                prompt += "\nSubtask examination from action perspective:"
                prompt += subtask
                prompt +="\nDomain file content:"
                prompt += domain_content
                prompt +="\n based on the objects availiable below."
                prompt += objects_ai 
                prompt += "Task description: extract out the problem files, based on the objects above, the precondition, actions and subtask examination."
                prompt += "#IMPORTANT, strictly follow the structure ,stop generate after the Problem file generation is done."


                if "gpt" not in args.gpt_version:
                # older versions of GPT
                    _, text = LM(prompt, args.gpt_version, max_tokens=1000, stop=["def"], frequency_penalty=0.30)
                else:            
                    # using variants of gpt 4o or 3.5
                    messages = [{"role": "system", "content": "You are a Robot PDDL problem Expert"},{"role": "user", "content": prompt}]
                    _, text = LM(messages, args.gpt_version, max_tokens=1400, frequency_penalty=0.4)

                problem_pddl.append(text)
                



            else:
                #modify to match other potential robots too.
                # Extract robot numbers and generate filename
                robot_pattern = re.compile(r"\s*\*{0,2}\s*robot\s*\??\s*(\d+)\*{0,2}", re.IGNORECASE)
                robot_numbers = robot_pattern.findall(assigned_robots)
                normalized_robot_numbers = [f"robot{num}" for num in robot_numbers]
                if robot_numbers:
                    robotassignnumber = f"{normalized_robot_numbers[0].replace(' ', '')}.pddl"
                    domain_path = os.path.join(os.getcwd(), "resources", robotassignnumber)
                    print("this is a solo work")
                    print(domain_path)
                    # Create full path for the filename

                    with open(domain_path, 'r') as file:
                        domain_content = file.read()
                        file.close()
                    problem_fileexamplepath = os.getcwd() + "/data/pythonic_plans/" + args.prompt_allocation_set + "_problem.py"
                    problem_fileexample = open(problem_fileexamplepath, "r")
                    problem_examplecontent = problem_fileexample.read()
                    problem_fileexample.close()

                    prompt = "\n"
                    prompt += problem_examplecontent
                    prompt = " Finish the tasks like example"
                    prompt += "\nSubtask examination from action perspective:"
                    prompt += subtask
                    prompt +="\nDomain file content:"
                    prompt += domain_content
                    prompt +="\n based on the objects availiable for potential usage below."
                    prompt += objects_ai 
                    prompt += "Task description: generate the problem file. Based on the objects above, the domain file precondition, actions and subtask examination. IMPORTANT the robot initate strictly as not inaction  and robot initate at robot (which includes location)"
                    prompt += "#IMPORTANT, strictly follow the structure ,stop generate after the Problem file generation is done."


                    if "gpt" not in args.gpt_version:
                    # older versions of GPT
                        _, text = LM(prompt, args.gpt_version, max_tokens=1000, stop=["def"], frequency_penalty=0.30)
                    else:            
                    # using variants of gpt 4o or 3.5
                        messages = [{"role": "system", "content": "You are a Robot PDDL problem Expert"},{"role": "user", "content": prompt}]
                        _, text = LM(messages, args.gpt_version, max_tokens=1400, frequency_penalty=0.4)

                    problem_pddl.append(text)
    print("the problem_pddl is")
    print(problem_pddl)
    return problem_pddl


def extract_domain_name(problem_file_path):
    # Extract the domain name from a problem PDDL file.
    domain_name = None
    domain_pattern = re.compile(r'\(\s*:domain\s+(\S+)\s*\)')
    
    with open(problem_file_path, 'r') as file:
        for line in file:
            # Remove any leading/trailing spaces from the line and search for the domain pattern
            match = domain_pattern.search(line.strip())
            if match:
                # Extract the domain name from the captured group
                domain_name = match.group(1)
                break

    return domain_name



def find_domain_file(base_path, domain_name):
    domain_path = os.path.join(base_path, domain_name + ".pddl")
    if os.path.isfile(domain_path):
        return domain_path
    return None

def parse_pddl_action(action_str):
    # Extracting action name and arguments
    action_pattern = re.compile(r'\((\w+)\s+(\w+)\s+(\w+)\s*(\w*)\)')
    match = action_pattern.match(action_str)
    
    if match:
        action_name = match.group(1).lower()  # Action name
        robot = match.group(2)  # First argument (robot)
        obj1 = match.group(3)  # Second argument (object or location)
        obj2 = match.group(4)  # Third argument (optional, location or target)

        # Mapping the action to a function name
        function_name = action_mapping.get(action_name, action_name)

        # Formatting the function call
        if obj2:
            return f"{function_name}({robot}, {obj1}, {obj2})"
        else:
            return f"{function_name}({robot}, {obj1})"
    return None


def extract_plan_from_output(content):
    plan_pattern = re.compile(r"^\s*\w+\s+\w+\s+\w+\s+\(\d+\)\s*$", re.MULTILINE)
    plan = plan_pattern.findall(content)
    
    # Joining the lines into a single string with newlines
    plan_output = "\n".join(plan)
    
    return plan
def match_the_references_forplan(combined_plan):
    prompt = ""
    prompt += objects_ai
    prompt += "IMPORTANT: Your TASK is based on the provided pddl plan provided in the passage below and the object list above, modify and only modify the plan so that all 'variablelocation' should be corrected to variable itself, since variable itself includes location. IMPORTANT: the only parenthesis usage should be for the correct PDDL plan, no exception."

    
    if "gpt" not in args.gpt_version:
         # older gpt versions
        _, text = LM(prompt, args.gpt_version, max_tokens=1000, stop=["def"], frequency_penalty=0.15)
    else:            
        messages = [{"role": "user", "content": prompt}]
        _, text = LM(messages,args.gpt_version, max_tokens=1300, frequency_penalty=0.0)

    code_planpddl.append(text)

    return code_planpddl

    
def save_code_planpddl_with_timestamp(code_planpddl):
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"code_planpddl_{timestamp}.txt"

    file_path = os.path.join(os.getcwd(), file_name)
    with open(file_path, "w") as file:
        for plan in code_planpddl:
            file.write(plan + "\n")


def save_combined_plan_with_timestamp(combined_plan):
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"combined_plan_{timestamp}.txt"

    file_path = os.path.join(os.getcwd(), file_name)
    with open(file_path, "w") as file:
        for plan in combined_plan:
            file.write(plan + "\n")

def combine_allplan_files():
    base_path = os.path.join(os.getcwd(), "resources", "generated_subtask")
    plan_files = [f for f in os.listdir(base_path) if f.endswith('_plan.txt')]

    prompt = ""

    for idx, filename in enumerate(plan_files):
        filepath = os.path.join(base_path, filename)
        with open(filepath, 'r') as infile:
            content = infile.read()
            plan = extract_plan_from_output(content)
            prompt += f"\nPlan {idx + 1}:\n{plan}\n"

    prompt += "allocation examination"
    prompt += sequence_operations
    prompt += "initial plan examination"
    for i, plan in enumerate(decomposed_plan):
        prompt = prompt + plan

    prompt += "you are robot allocation expert, Your task is, based on inital plan examination and allocation examination correct the subplans. Then based on your understanding merge the subtasks together by using timed durative actions format, where parallel tasks are performed at the same time. IMPORTANT: all 'variablelocation' should be corrected to variable itself, since variable itself includes location. and result must be in PDDL plan format."

        

    print("prompt is")
    print(prompt)
    if "gpt" not in args.gpt_version:
         # older gpt versions
        _, text = LM(prompt, args.gpt_version, max_tokens=1000, stop=["def"], frequency_penalty=0.15)
    else:            
        messages = [{"role": "user", "content": prompt}]
        _, text = LM(messages,args.gpt_version, max_tokens=1300, frequency_penalty=0.0)

    
    
    combined_plan.append(text)

    
    return combined_plan

def validate_subplan_into_prompt():
    base_path = os.path.join(os.getcwd(), "resources", "generated_subtask")
    plan_files = [f for f in os.listdir(base_path) if f.endswith('_plan.txt')]



def clean_directory(directory_path):

    if os.path.exists(directory_path):

        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)

            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove the file or link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--floor-plan", type=int, required=True)
    parser.add_argument("--openai-api-key-file", type=str, default="api_key")
    parser.add_argument("--gpt-version", type=str, default="gpt-4o", 
                        choices=['gpt-3.5-turbo', 'gpt-4o', 'gpt-3.5-turbo-16k'])
    
    parser.add_argument("--prompt-decompse-set", type=str, default="pddl_train_task_decomposesep", 
                        choices=['pddl_train_task_decompose'])
    
    parser.add_argument("--prompt-allocation-set", type=str, default="pddl_train_task_allocationsep", 
                        choices=['pddl_train_task_allocation'])
    
    parser.add_argument("--test-set", type=str, default="final_test", 
                        choices=['final_test'])
    
    parser.add_argument("--log-results", type=bool, default=True)
    
    args = parser.parse_args()

    set_api_key(args.openai_api_key_file)

    # set path to clean temp generatedfile, refer to log file instead.
    directory_to_clean = os.path.join(os.getcwd(), "resources", "generated_subtask")
    clean_directory(directory_to_clean)

    if not os.path.isdir(f"./logs/"):
        os.makedirs(f"./logs/")
        
    # read the tasks        
    test_tasks = []
    robots_test_tasks = []  
    gt_test_tasks = []    
    trans_cnt_tasks = []
    min_trans_cnt_tasks = []  
    with open (f"./data/{args.test_set}/FloorPlan{args.floor_plan}.json", "r") as f:
        for line in f.readlines():
            test_tasks.append(list(json.loads(line).values())[0])
            robots_test_tasks.append(list(json.loads(line).values())[1])
            gt_test_tasks.append(list(json.loads(line).values())[2])
            trans_cnt_tasks.append(list(json.loads(line).values())[3])
            min_trans_cnt_tasks.append(list(json.loads(line).values())[4])
                    
    print(f"\n----Test set tasks----\n{test_tasks}\nTotal: {len(test_tasks)} tasks\n")
    # prepare list of robots for the tasks
    available_robots = []
    for robots_list in robots_test_tasks:
        task_robots = []
        for i, r_id in enumerate(robots_list):
            rob = robots.robots [r_id-1]
            # rename the robot
            rob['name'] = 'robot' + str(i+1)
            task_robots.append(rob)
        available_robots.append(task_robots)
        
    #print(available_robots) 

    ######## Train Task Decomposition into separate single tasks########
         
    # prepare train decompostion demonstration for ai2thor samples
    allaction_pddldomain = open(os.getcwd() + "/resources/allactionrobot.pddl","r")
    prompt = f"from pddl domain file with all possible actions: " +"\n" + allaction_pddldomain.read()
    objects_ai = f"\n\nobjects = {get_ai2_thor_objects(args.floor_plan)}"
    prompt += objects_ai
     
    # read input train prompts
    decompose_prompt_file = open(os.getcwd() + "/data/pythonic_plans/" + args.prompt_decompse_set + ".py", "r")
    decompose_prompt = decompose_prompt_file.read()
    decompose_prompt_file.close()
    
    prompt += "robot initaite 'as not inaction robot '(which defaults location too)\n\n" + decompose_prompt
    prompt += "# GENERAL TASK DECOMPOSITION \n Decompose and parallel subtasks where ever possible."
    
    print ("Generating Decompsed Plans...")
    
    decomposed_plan = []
    for task in test_tasks:
        prompt =  f"{prompt}\n\n# Task Description: {task}"
        
        if "gpt" not in args.gpt_version:
            # older gpt versions
            _, text = LM(prompt, args.gpt_version, max_tokens=1000, stop=["def"], frequency_penalty=0.15)
        else:            
            messages = [{"role": "user", "content": prompt}]
            _, text = LM(messages,args.gpt_version, max_tokens=1300, frequency_penalty=0.0)

        decomposed_plan.append(text)


    print("decompose done, wait 50sec for api calling allocation solution")
    #print("the decompsed_plan is")
    #print(decomposed_plan)
    #time.sleep(65) 



    print ("Generating Allocation plan Solution...")



    ### switch generation###￥
 



###  Train task allocation- solution ###


    
    
    
    prompt_file = os.getcwd() + "/data/pythonic_plans/" + args.prompt_allocation_set + "_solution.py"
    allocated_prompt_file = open(prompt_file, "r")
    allocated_prompt = allocated_prompt_file.read()
    allocated_prompt_file.close()
    
    prompt = "\n"
    prompt += allocated_prompt
    
    allocated_plan = []
    for i, plan in enumerate(decomposed_plan):
        no_robot  = len(available_robots[i])
        prompt = prompt + plan
        prompt += f"\n# TASK ALLOCATION"
        prompt += f"\n# Scenario: There are {no_robot} robots available. The task should be performed using the minimum number of robots necessary. Robot should be assigned to subtasks that match its skills and mass capacity. Using your reasoning come up with a solution to satisfy all contraints."
        prompt += f"\n\nrobots = {available_robots[i]}"
        prompt += f"\n{objects_ai}"
        prompt += f"\n\n# IMPORTANT: The AI should ensure that the robots assigned to the tasks have all the necessary skills to perform the tasks. IMPORTANT: Determine whether the subtasks must be performed sequentially or in parallel, or a combination of both and allocate robots based on availablitiy. "
        prompt += f"\n# SOLUTION  \n"
        #print("prompt for allocation plan is")
        #print(prompt)
    
        if "gpt" not in args.gpt_version:
            # older versions of GPT
            _, text = LM(prompt, args.gpt_version, max_tokens=1000, stop=["def"], frequency_penalty=0.65)
        
        elif "gpt-3.5" in args.gpt_version:
            # gpt 3.5 and its variants
            messages = [{"role": "user", "content": prompt}]
            _, text = LM(messages, args.gpt_version, max_tokens=1500, frequency_penalty=0.35)
        
        else:          
            # gpt 4.0o
            messages = [{"role": "user", "content": prompt}]
            _, text = LM(messages, args.gpt_version, max_tokens=400, frequency_penalty=0.69)

        allocated_plan.append(text)

    #print("the allocation_plan is")
    #print(allocated_plan)

        

 
    print("allocation done, wait 60sec for allocated content summary")
    #time.sleep(60) 

    print ("Generating Allocated solutions...")    

### align problem file domain with the choosen robot
    prompt = " finish the problem content summary strictly following the example format"
     
    code_plan = []

    prompt_file1 = os.getcwd() + "/data/pythonic_plans/" + args.prompt_allocation_set + "_summary.py"
    code_prompt_file = open(prompt_file1, "r")
    code_prompt = code_prompt_file.read()
    code_prompt_file.close()
    
    prompt += "\n\n" + code_prompt + "\n\n"

    for i, (plan, solution) in enumerate(zip(decomposed_plan,allocated_plan)):
        prompt = prompt + plan
        prompt += f"\n# TASK ALLOCATION"
        prompt += f"\n\nrobots = {available_robots[i]}"
        prompt += solution
        prompt += f"\n# problem content summary  \n"
        
        if "gpt" not in args.gpt_version:
            # older versions of GPT
            _, text = LM(prompt, args.gpt_version, max_tokens=1000, stop=["def"], frequency_penalty=0.30)
        else:            
            # using variants of gpt 4 or 3.5
            messages = [{"role": "system", "content": "You are a Robot PDDL problem Expert"},{"role": "user", "content": prompt}]
            _, text = LM(messages, args.gpt_version, max_tokens=1400, frequency_penalty=0.4)

        code_plan.append(text)
        
    print("Problem summary done, wait 60sec to generate problem file for each subtasks")
    #print("code_plan is")
    #print(code_plan)
    
    
    #Generating problem file
    for i,plan in enumerate(code_plan):
        problem_pddl = []
        
        
        subtasks, sequence_operations = split_and_store_tasks(plan) #split subtasks and sequence_operations
        #print("the subtasks are")
        #print(subtasks)
        problemextracting(subtasks)



    print("allproblem file pddl list:")
    print(problem_pddl)









    #time.sleep(65)
    #print("wait for cleaning codeplan")

   # cleaned_code_plan = []
   # for i, plan in enumerate(code_plan):
    #    prompt = "process the following content,IMPORTANT,  Clear all the other symbols and words outside of each pddl problem file left only the unmodified subtask problem files, and at the very beginning start your respond with '###subtaskdevideline##'"
    #    prompt = prompt + plan
     #   print(prompt)


       # if "gpt" not in args.gpt_version:
            
       #     _, text = LM(prompt, args.gpt_version, max_tokens=1000, stop=["def"], frequency_penalty=0.30)
       # else:            

        #    messages = [{"role": "system", "content": "You are a Robot PDDL problem Expert"},{"role": "user", "content": prompt}]
#            _, text = LM(messages, args.gpt_version, max_tokens=1400, frequency_penalty=0.4)
            
        

 #       cleaned_code_plan.append(text)

        #print(cleaned_code_plan)


    #print(code_plan)

    #timestamp folder
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    timestamp_directory = os.path.join(os.getcwd(), "resources", "each_run", timestamp)   

    split_pddl_tasks(problem_pddl)
    print("wait 50sec for api calling llmvalidator")


    run_llmvalidator()
    run_planners()



    combined_plan = []
    #validate and generate a merged plan

    combine_allplan_files()
    save_combined_plan_with_timestamp(combined_plan)

    code_planpddl = []
    match_the_references_forplan(combined_plan)
    save_code_planpddl_with_timestamp(code_planpddl)

    #



#log results
    exec_folders = []
    if args.log_results:
        line = {}
        now = datetime.now() # current date and time
        date_time = now.strftime("%m-%d-%Y-%H-%M-%S")
        
        for idx, task in enumerate(test_tasks):
            TC, total_subtasks = calculate_task_completion_rate()

            task_name = "{fxn}".format(fxn = '_'.join(task.split(' ')))
            task_name = task_name.replace('\n','')
            folder_name = f"{task_name}_plans_{date_time}"
            exec_folders.append(folder_name)
            
            os.mkdir("./logs/"+folder_name)
     
            with open(f"./logs/{folder_name}/log.txt", 'w') as f:
                f.write(task)
                f.write(f"\n\nGPT Version: {args.gpt_version}")
                f.write(f"\n\nFloor Plan: {args.floor_plan}")
                f.write(f"\n{objects_ai}")
                f.write(f"\nrobots = {available_robots[idx]}")
                f.write(f"\nground_truth = {gt_test_tasks[idx]}")
                f.write(f"\ntrans = {trans_cnt_tasks[idx]}")
                f.write(f"\nmin_trans = {min_trans_cnt_tasks[idx]}")
                f.write(f"\nTotalsuccesssubtask = {TC}")
                f.write(f"\nTotalsubtask = {total_subtasks}")
                #f.write(f"\nCurrentTrans = {}")

            with open(f"./logs/{folder_name}/code_planpddl.py", 'w') as x:
                x.write(code_planpddl[idx])

            with open(f"./logs/{folder_name}/combined_plan.py",'w') as k:
                k.write(combined_plan[idx])

            with open(f"./logs/{folder_name}/decomposed_plan.py", 'w') as d:
                d.write(decomposed_plan[idx])
                
            with open(f"./logs/{folder_name}/allocated_plan.py", 'w') as a:
                a.write(allocated_plan[idx])
                
            with open(f"./logs/{folder_name}/code_plan.py", 'w') as x:
                x.write(code_plan[idx])

            subtask_folder = f"./logs/{folder_name}/generated_subtask"
            os.mkdir(subtask_folder)

            source_folder = "./resources/generated_subtask"
            for file_name in os.listdir(source_folder):
                full_file_name = os.path.join(source_folder, file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, subtask_folder)





























#####



    




