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
from typing import List, Dict, Tuple, Optional

import openai
import ai2thor.controller

import sys
sys.path.append(".")

import resources.actions as actions
import resources.robots as robots

# Constants
DEFAULT_MAX_TOKENS = 128
DEFAULT_TEMPERATURE = 0
DEFAULT_RETRY_DELAY = 20
MAX_RETRIES = 3

# Custom Exceptions
class PDDLError(Exception):
    """Base exception for PDDL-related errors"""
    pass

class ValidationError(PDDLError):
    """Raised when PDDL validation fails"""
    pass

class PlanningError(PDDLError):
    """Raised when PDDL planning fails"""
    pass

class LLMError(Exception):
    """Raised when there are issues with LLM operations"""
    pass

class LLMHandler:
    """Handles all interactions with the language model"""
    
    def __init__(self, api_key_file: str):
        """Initialize the LLM handler with API key file.
        
        Args:
            api_key_file (str): Path to the API key file (without .txt extension)
        """
        self.setup_api(api_key_file)
    
    def setup_api(self, api_key_file: str) -> None:
        """Set up the OpenAI API key.
        
        Args:
            api_key_file (str): Path to the API key file (without .txt extension)
        
        Raises:
            LLMError: If there's an error reading the API key file
        """
        try:
            openai.api_key = Path(api_key_file + '.txt').read_text()
        except FileNotFoundError:
            raise LLMError(f"API key file {api_key_file}.txt not found")
        except Exception as e:
            raise LLMError(f"Error reading API key file: {str(e)}")
    
    def query_model(
        self, 
        prompt: str | List[Dict], 
        gpt_version: str, 
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        stop: Optional[List[str]] = None,
        logprobs: Optional[int] = 1,
        frequency_penalty: float = 0
    ) -> Tuple[dict, str]:
        """Query the language model with retry mechanism.
        
        Args:
            prompt: Either a string (for older models) or a list of message dicts (for chat models)
            gpt_version: The model version to use
            max_tokens: Maximum number of tokens in the response
            temperature: Sampling temperature
            stop: Optional list of stop sequences
            logprobs: Optional number of logprobs to return
            frequency_penalty: Frequency penalty for token generation
        
        Returns:
            Tuple of (full response object, generated text)
            
        Raises:
            LLMError: If all retries fail
        """
        retry_delay = DEFAULT_RETRY_DELAY
        
        for attempt in range(MAX_RETRIES):
            try:
                if "gpt" not in gpt_version:
                    response = openai.Completion.create(
                        model=gpt_version, 
                        prompt=prompt, 
                        max_tokens=max_tokens, 
                        temperature=temperature, 
                        stop=stop, 
                        logprobs=logprobs, 
                        frequency_penalty=frequency_penalty
                    )
                    return response, response["choices"][0]["text"].strip()
                else:
                    response = openai.ChatCompletion.create(
                        model=gpt_version, 
                        messages=prompt, 
                        max_tokens=max_tokens, 
                        temperature=temperature, 
                        frequency_penalty=frequency_penalty
                    )
                    return response, response["choices"][0]["message"]["content"].strip()
                    
            except openai.error.RateLimitError:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                raise LLMError("Rate limit exceeded after all retries")
                
            except (openai.error.APIError, openai.error.Timeout) as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(retry_delay)
                    continue
                raise LLMError(f"API Error after all retries: {str(e)}")
                
            except Exception as e:
                raise LLMError(f"Unexpected error in LLM query: {str(e)}")

# Function returns object list with name and properties.
def convert_to_dict_objprop(objs, obj_mass):
    objs_dict = []
    for i, obj in enumerate(objs):
        obj_dict = {'name': obj , 'mass' : obj_mass[i]}
        # obj_dict = {'name': obj , 'mass' : 1.0}
        objs_dict.append(obj_dict)
    return objs_dict

"""
def get_ai2_thor_objects(floor_plan_id):
    # connector to ai2thor to get object list
    controller = ai2thor.controller.Controller(scene="FloorPlan"+str(floor_plan_id))
    obj = list([obj["objectType"] for obj in controller.last_event.metadata["objects"]])
    obj_mass = list([obj["mass"] for obj in controller.last_event.metadata["objects"]])
    controller.stop()
    obj = convert_to_dict_objprop(obj, obj_mass)
    return obj
"""

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
    try:
        directory = os.path.join(os.getcwd(), "resources", "generated_subtask")
        os.makedirs(directory, exist_ok=True)

        for i, plan in enumerate(code_plan):
            tasks = re.split(r"\s*\(define\s*\(problem", plan)

            for j, task in enumerate(tasks[1:]):
                print("current task is:")
                print(task)
                task = "(define (problem" + task
                task = balance_parentheses(task)
                
                match = re.search(r'\(problem\s+(\w+)\)', task, re.IGNORECASE)

                if match:
                    task_name = match.group(1)
                    filename = f"{i+1}_{j+1}_{task_name}.pddl"
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"{i+1}_{j+1}_{timestamp}.pddl"

                filepath = os.path.join(directory, filename)
                
                try:
                    with open(filepath, 'w') as file:
                        file.write(task)
                except Exception as e:
                    print(f"Error writing task to file {filepath}: {str(e)}")
                    continue
    except Exception as e:
        print(f"Error in split_pddl_tasks: {str(e)}")
        raise

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
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading file {file_path}: {str(e)}")

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

class FileProcessor:
    """Handles file operations and text processing"""
    
    def __init__(self, base_path: str):
        """Initialize the file processor.
        
        Args:
            base_path (str): Base path for all file operations
        """
        self.base_path = base_path
        self.subtask_path = os.path.join(base_path, "resources", "generated_subtask")
        os.makedirs(self.subtask_path, exist_ok=True)
    
    def read_file(self, file_path: str) -> str:
        """Read contents of a file.
        
        Args:
            file_path (str): Path to the file to read
            
        Returns:
            str: Contents of the file
            
        Raises:
            PDDLError: If there's an error reading the file
        """
        try:
            with open(file_path, 'r') as file:
                return file.read()
        except FileNotFoundError:
            raise PDDLError(f"File not found: {file_path}")
        except Exception as e:
            raise PDDLError(f"Error reading file {file_path}: {str(e)}")
    
    def write_file(self, file_path: str, content: str) -> None:
        """Write content to a file.
        
        Args:
            file_path (str): Path to the file to write
            content (str): Content to write to the file
            
        Raises:
            PDDLError: If there's an error writing to the file
        """
        try:
            with open(file_path, 'w') as file:
                file.write(content)
        except Exception as e:
            raise PDDLError(f"Error writing to file {file_path}: {str(e)}")
    
    def split_pddl_tasks(self, code_plan: List[str]) -> None:
        """Split PDDL tasks and save them to files.
        
        Args:
            code_plan (List[str]): List of PDDL task descriptions
            
        Raises:
            PDDLError: If there's an error processing the tasks
        """
        try:
            for i, plan in enumerate(code_plan):
                tasks = re.split(r"\s*\(define\s*\(problem", plan)
                
                for j, task in enumerate(tasks[1:]):
                    task = "(define (problem" + task
                    task = self.balance_parentheses(task)
                    
                    match = re.search(r'\(problem\s+(\w+)\)', task, re.IGNORECASE)
                    if match:
                        task_name = match.group(1)
                        filename = f"{i+1}_{j+1}_{task_name}.pddl"
                    else:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = f"{i+1}_{j+1}_{timestamp}.pddl"
                    
                    filepath = os.path.join(self.subtask_path, filename)
                    self.write_file(filepath, task)
                    
        except Exception as e:
            raise PDDLError(f"Error splitting PDDL tasks: {str(e)}")
    
    def balance_parentheses(self, content: str) -> str:
        """Balance parentheses in PDDL content.
        
        Args:
            content (str): PDDL content to process
            
        Returns:
            str: Processed PDDL content with balanced parentheses
        """
        open_count = 0
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
            return content[start_index:end_index+1]
        return ""
    
    def split_and_store_tasks(self, content: str) -> Tuple[List[str], str]:
        """Split and store tasks from content.
        
        Args:
            content (str): Content to process
            
        Returns:
            Tuple[List[str], str]: Tuple of (subtasks list, sequence operations)
        """
        summary_pattern = re.compile(r'#?\s*Problem content summary\s*:?(.*?)(?=#?\s*Sequence of Operations?\s*:?)', re.DOTALL | re.IGNORECASE)
        sequence_pattern = re.compile(r'#?\s*Sequence of Operations?\s*:\s*(.*)', re.DOTALL | re.IGNORECASE)
        
        summary_match = summary_pattern.search(content)
        problem_summary = summary_match.group(1).strip() if summary_match else "failed to extract1"
        
        sequence_match = sequence_pattern.search(content)
        sequence_operations = sequence_match.group(1).strip() if sequence_match else "failed to extract2"
        
        subtasks = re.split(r'\s*\w?\s*(?:\n)?\s*(?=#\s*subtask\s*:?)\s*', problem_summary, flags=re.IGNORECASE)
        subtasks = [subtask.strip() for subtask in subtasks if subtask.strip()]
        
        return subtasks, sequence_operations

class PDDLValidator:
    """Handles PDDL validation operations"""
    
    def __init__(self, llm_handler: LLMHandler, file_processor: FileProcessor):
        """Initialize the PDDL validator.
        
        Args:
            llm_handler (LLMHandler)
            file_processor (FileProcessor)
        """
        self.llm = llm_handler
        self.file_processor = file_processor
    
    def validate_problem(self, domain_file: str, problem_file: str, gpt_version: str) -> None:
        """Validate a PDDL problem file against its domain.
        
        Args:
            domain_file (str)
            problem_file (str)
            gpt_version (str)

        """
        try:
            domain_content = self.file_processor.read_file(domain_file)
            problem_content = self.file_processor.read_file(problem_file)
            
            prompt = (
                f"Domain Description:\n{domain_content}\n\n"
                f"Problem Description:\n{problem_content}\n\n"
                "Validate the preconditions in problem file to ensure all precondition listed object "

            )
            
            if "gpt" not in gpt_version:
                _, validated_text = self.llm.query_model(
                    prompt=prompt,
                    gpt_version=gpt_version,
                    max_tokens=1000,
                    stop=["def"],
                    frequency_penalty=0.30
                )
            else:
                messages = [
                    {"role": "system", "content": "You are a Robot PDDL problem Expert"},
                    {"role": "user", "content": prompt}
                ]
                _, validated_text = self.llm.query_model(
                    prompt=messages,
                    gpt_version=gpt_version,
                    max_tokens=1400,
                    frequency_penalty=0.4
                )
            
            # Save the validated content back to the problem file
            self.file_processor.write_file(problem_file, validated_text)
            
        except Exception as e:
            raise ValidationError(f"Error validating PDDL problem: {str(e)}")

class PDDLPlanner:
    
    def __init__(self, base_path: str, file_processor: FileProcessor):
        """Initialize the PDDL planner.
        
        Args:
            base_path (str)
            file_processor (FileProcessor)
        """
        self.base_path = base_path
        self.file_processor = file_processor
        self.planner_path = os.path.join(base_path, "downward", "fast-downward.py")
    
    def run_plan(self, domain_file: str, problem_file: str) -> None:
        """
        
        Args:
            domain_file (str)
            problem_file (str)
    
        """
        try:
            command = [
                self.planner_path,
                "--alias",
                "seq-opt-lmcut",
                domain_file,
                problem_file
            ]
            
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Save the plan output
            output_file = problem_file.replace('.pddl', '_plan.txt')
            self.file_processor.write_file(output_file, result.stdout)
            
            if result.stderr:
                print(f"Warnings/Errors for {problem_file}:", result.stderr)
                
        except Exception as e:
            raise PlanningError(f"Error running PDDL planner: {str(e)}")
    
    def calculate_completion_rate(self) -> Tuple[int, int]:
        """
        
        Returns:
            Tuple[int, int]: (number of completed tasks, total number of tasks)
        """
        TC = 0
        total_subtasks = 0
        
        try:
            for file_path in glob.glob(os.path.join(self.file_processor.subtask_path, '*_plan.txt')):
                total_subtasks += 1
                content = self.file_processor.read_file(file_path)
                TC += content.count('Solution found!')
                
            return TC, total_subtasks
            
        except Exception as e:
            raise PlanningError(f"Error calculating completion rate: {str(e)}")

class TaskManager:
    """Manages task processing and coordination"""
    
    def __init__(self, base_path: str, gpt_version: str, api_key_file: str):
        """Initialize the task manager.
        
        Args:
            base_path (str): Base path for operations
            gpt_version (str): Version of GPT to use
            api_key_file (str): Path to the API key file
        """
        self.base_path = base_path
        self.gpt_version = gpt_version
        
        # Initialize components
        self.llm = LLMHandler(api_key_file)
        self.file_processor = FileProcessor(base_path)
        self.validator = PDDLValidator(self.llm, self.file_processor)
        self.planner = PDDLPlanner(base_path, self.file_processor)
        
        # Initialize paths
        self.resources_path = os.path.join(base_path, "resources")
    
    def process_tasks(self, test_tasks: List[str], available_robots: List[dict]) -> None:
        """Process a list of tasks.
        
        Args:
            test_tasks (List[str]): List of tasks to process
            available_robots (List[dict]): List of available robots for tasks
        """
        try:
            # Get domain content
            allaction_domain_path = os.path.join(self.resources_path, "allactionrobot.pddl")
            domain_content = self.file_processor.read_file(allaction_domain_path)
            
            # Process each task
            for task_idx, (task, robots) in enumerate(zip(test_tasks, available_robots)):
                print(f"\nProcessing task {task_idx + 1}: {task}")
                
                # Generate decomposed plan
                decomposed_plan = self._generate_decomposed_plan(task, domain_content, robots)
                
                # Generate allocation plan
                allocated_plan = self._generate_allocation_plan(decomposed_plan, robots)
                
                # Generate problem files
                code_plan = self._generate_problem_files(decomposed_plan, allocated_plan, robots)
                
                # Split into subtasks
                self.file_processor.split_pddl_tasks(code_plan)
                
                # Validate and plan
                self._validate_and_plan()
                
                # Calculate completion rate
                tc, total = self.planner.calculate_completion_rate()
                print(f"Task completion rate: {tc}/{total}")
                
        except Exception as e:
            print(f"Error processing tasks: {str(e)}")
            raise
    
    def _generate_decomposed_plan(self, task: str, domain_content: str, robots: List[dict]) -> str:
        """Generate decomposed plan for a task."""
        prompt = (
            f"from pddl domain file with all possible actions: \n{domain_content}\n\n"
            f"robots = {robots}\n\n"
            "robot initiate 'as not inaction robot '(which defaults location too)\n\n"
            "# GENERAL TASK DECOMPOSITION \n"
            "Decompose and parallel subtasks where ever possible.\n\n"
            f"# Task Description: {task}"
        )
        
        if "gpt" not in self.gpt_version:
            _, text = self.llm.query_model(prompt, self.gpt_version, max_tokens=1000, stop=["def"], frequency_penalty=0.15)
        else:
            messages = [{"role": "user", "content": prompt}]
            _, text = self.llm.query_model(messages, self.gpt_version, max_tokens=1300, frequency_penalty=0.0)
        
        return text
    
    def _generate_allocation_plan(self, decomposed_plan: str, robots: List[dict]) -> str:
        """Generate allocation plan for decomposed tasks."""
        prompt = (
            f"{decomposed_plan}\n"
            f"# TASK ALLOCATION\n"
            f"# Scenario: There are {len(robots)} robots available. "
            "The task should be performed using the minimum number of robots necessary. "
            "Robot should be assigned to subtasks that match its skills and mass capacity. "
            "Using your reasoning come up with a solution to satisfy all constraints.\n\n"
            f"robots = {robots}\n\n"
            "# IMPORTANT: The AI should ensure that the robots assigned to the tasks have "
            "all the necessary skills to perform the tasks. IMPORTANT: Determine whether "
            "the subtasks must be performed sequentially or in parallel, or a combination "
            "of both and allocate robots based on availability.\n"
            "# SOLUTION\n"
        )
        
        if "gpt" not in self.gpt_version:
            _, text = self.llm.query_model(prompt, self.gpt_version, max_tokens=1000, stop=["def"], frequency_penalty=0.65)
        else:
            messages = [{"role": "user", "content": prompt}]
            _, text = self.llm.query_model(messages, self.gpt_version, max_tokens=1500, frequency_penalty=0.35)
        
        return text
    
    def _generate_problem_files(self, decomposed_plan: str, allocated_plan: str, robots: List[dict]) -> List[str]:
        """Generate PDDL problem files from plans."""
        prompt = (
            f"{decomposed_plan}\n"
            f"# TASK ALLOCATION\n"
            f"robots = {robots}\n"
            f"{allocated_plan}\n"
            "# problem content summary\n"
        )
        
        if "gpt" not in self.gpt_version:
            _, text = self.llm.query_model(prompt, self.gpt_version, max_tokens=1000, stop=["def"], frequency_penalty=0.30)
        else:
            messages = [
                {"role": "system", "content": "You are a Robot PDDL problem Expert"},
                {"role": "user", "content": prompt}
            ]
            _, text = self.llm.query_model(messages, self.gpt_version, max_tokens=1400, frequency_penalty=0.4)
        
        return [text]
    
    def _validate_and_plan(self) -> None:
        """Validate and plan all problem files."""
        problem_files = [
            f for f in os.listdir(self.file_processor.subtask_path)
            if f.endswith('.pddl')
        ]
        
        for problem_file in problem_files:
            problem_path = os.path.join(self.file_processor.subtask_path, problem_file)
            domain_name = self._extract_domain_name(problem_path)
            
            if not domain_name:
                print(f"No domain specified in {problem_file}")
                continue
            
            domain_file = self._find_domain_file(domain_name)
            if not domain_file:
                print(f"No domain file found for domain {domain_name}")
                continue
            
            # Validate and plan
            self.validator.validate_problem(domain_file, problem_path, self.gpt_version)
            self.planner.run_plan(domain_file, problem_path)
    
    def _extract_domain_name(self, problem_file: str) -> Optional[str]:
        """Extract domain name from problem file."""
        try:
            content = self.file_processor.read_file(problem_file)
            match = re.search(r'\(:domain\s+(\w+)\)', content)
            return match.group(1) if match else None
        except Exception:
            return None
    
    def _find_domain_file(self, domain_name: str) -> Optional[str]:
        """Find domain file by name."""
        domain_path = os.path.join(self.resources_path, f"{domain_name}.pddl")
        return domain_path if os.path.isfile(domain_path) else None

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--floor-plan", type=int, required=True)
    parser.add_argument("--openai-api-key-file", type=str, default="api_key")
    parser.add_argument(
        "--gpt-version",
        type=str,
        default="gpt-4o",
        choices=['gpt-3.5-turbo', 'gpt-4o', 'gpt-3.5-turbo-16k']
    )
    parser.add_argument(
        "--prompt-decompse-set",
        type=str,
        default="pddl_train_task_decomposesep",
        choices=['pddl_train_task_decompose']
    )
    parser.add_argument(
        "--prompt-allocation-set",
        type=str,
        default="pddl_train_task_allocationsep",
        choices=['pddl_train_task_allocation']
    )
    parser.add_argument(
        "--test-set",
        type=str,
        default="final_test",
        choices=['final_test']
    )
    parser.add_argument("--log-results", type=bool, default=True)
    
    return parser.parse_args()

def main():
    """Main execution function."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Initialize task manager
        task_manager = TaskManager(
            base_path=os.getcwd(),
            gpt_version=args.gpt_version,
            api_key_file=args.openai_api_key_file
        )
        
        # Load test tasks and robots
        test_tasks = []
        available_robots = []
        
        test_file = os.path.join(
            "data",
            args.test_set,
            f"FloorPlan{args.floor_plan}.json"
        )
        
        with open(test_file, "r") as f:
            for line in f:
                data = json.loads(line)
                test_tasks.append(list(data.values())[0])
                robots_list = list(data.values())[1]
                
                # Prepare robots
                task_robots = []
                for i, r_id in enumerate(robots_list):
                    rob = robots.robots[r_id-1].copy()
                    rob['name'] = f'robot{i+1}'
                    task_robots.append(rob)
                available_robots.append(task_robots)
        
        print(f"\n----Test set tasks----\n{test_tasks}\nTotal: {len(test_tasks)} tasks\n")
        
        # Process tasks
        task_manager.process_tasks(test_tasks, available_robots)
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()





























#####



    




