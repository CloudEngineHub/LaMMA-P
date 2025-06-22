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
from typing import List, Dict, Tuple, Optional, Union, Any

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

# Action mapping from actions module

class PDDLError(Exception):
    """Base exception class for PDDL-related errors."""
    pass

class ValidationError(PDDLError):
    """Exception raised for PDDL validation errors."""
    pass

class PlanningError(PDDLError):
    """Exception raised for PDDL planning errors."""
    pass

class LLMError(Exception):
    """Exception raised for Language Model related errors."""
    pass

class PDDLUtils:
    """Utility functions for PDDL operations."""
    
    @staticmethod
    def convert_to_dict_objprop(objs: List[str], obj_mass: List[float]) -> List[Dict[str, Union[str, float]]]:
        """Convert object list to dictionary format with mass.
        
        Args:
            objs (List[str]): List of object names
            obj_mass (List[float]): List of object masses
            
        Returns:
            List[Dict[str, Union[str, float]]]: List of dictionaries containing object properties
        """
        return [{'name': obj, 'mass': mass} for obj, mass in zip(objs, obj_mass)]
    
    @staticmethod
    def get_ai2_thor_objects(floor_plan: int) -> List[Dict[str, Any]]:
        """Get objects from AI2Thor environment.
        
        Args:
            floor_plan (int): Floor plan number
            
        Returns:
            List[Dict[str, Any]]: List of objects with their properties
        """
        controller = None
        try:
            controller = ai2thor.controller.Controller(scene=f"FloorPlan{floor_plan}")
            obj = [obj["objectType"] for obj in controller.last_event.metadata["objects"]]
            obj_mass = [obj["mass"] for obj in controller.last_event.metadata["objects"]]
            return PDDLUtils.convert_to_dict_objprop(obj, obj_mass)
        finally:
            if controller:
                controller.stop()

class FileProcessor:
    """Handles file operations and text processing for PDDL files.
    
    This class manages reading, writing, and processing of PDDL files and related
    text content. It provides methods for file operations and text manipulation
    specific to PDDL task processing.
    """
    
    def __init__(self, base_path: str):
        """Initialize the file processor.
        
        Args:
            base_path (str): Base path for file operations
        """
        self.base_path = base_path
        self.subtask_path = os.path.join(base_path, "resources", "generated_subtask")
        self.each_run_path = os.path.join(base_path, "resources", "each_run")
        os.makedirs(self.subtask_path, exist_ok=True)
        os.makedirs(self.each_run_path, exist_ok=True)
    
    def read_file(self, file_path: str) -> str:
        """Read contents of a file.
        
        Args:
            file_path (str): Path to the file to read
            
        Returns:
            str: Contents of the file
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
            file_path (str): Path to write to
            content (str): Content to write
        """
        try:
            with open(file_path, 'w') as file:
                file.write(content)
        except Exception as e:
            raise PDDLError(f"Error writing to file {file_path}: {str(e)}")
    
    def split_pddl_tasks(self, code_plan: Union[str, List[str]]) -> None:
        """Split PDDL tasks and save them to files.
        
        Args:
            code_plan (List[str]): List of PDDL plans to split
        """
        try:
            # Convert single plan to list
            if isinstance(code_plan, str):
                code_plan = [code_plan]
            
            # Create timestamped directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamp_directory = os.path.join(self.each_run_path, timestamp)
            os.makedirs(timestamp_directory, exist_ok=True)
            
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
                    
                    # Save to both timestamped directory and generated_subtask
                    filepath = os.path.join(timestamp_directory, filename)
                    self.write_file(filepath, task)
                    
                    # Also save to generated_subtask for compatibility
                    subtask_filepath = os.path.join(self.subtask_path, filename)
                    self.write_file(subtask_filepath, task)
            
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
    
    def split_and_store_tasks(self, content: str, llm: Optional['LLMHandler'] = None, gpt_version: Optional[str] = None) -> Tuple[List[str], str]:
        """Split and store tasks from content.
        
        Returns:
            Tuple[List[str], str]: Tuple of (subtasks list, sequence operations)
        """
        # More flexible pattern for summary and sequence
        summary_pattern = re.compile(r'(?:#?\s*)?Problem\s*content\s*summary\s*:?(.*?)(?=(?:#?\s*)?Sequence\s*of\s*Operations?\s*:?)', re.DOTALL | re.IGNORECASE)
        sequence_pattern = re.compile(r'(?:#?\s*)?Sequence\s*of\s*Operations?\s*:\s*(.*)', re.DOTALL | re.IGNORECASE)
        
        summary_match = summary_pattern.search(content)
        problem_summary = summary_match.group(1).strip() if summary_match else "failed to extract1"
        
        sequence_match = sequence_pattern.search(content)
        sequence_operations = sequence_match.group(1).strip() if sequence_match else "failed to extract2"
        
        # More flexible pattern for splitting subtasks
        subtasks = re.split(r'(?:#?\s*)?subtask\s*:?\s*', problem_summary, flags=re.IGNORECASE)
        subtasks = [subtask.strip() for subtask in subtasks if subtask.strip()]
        
        # Fix structure of each subtask if needed
        fixed_subtasks = []
        for subtask in subtasks:
            # Pattern to match the format expected by problemextracting
            pattern = re.compile(
                r'\s*\*{0,2}\s*Assigned\s*Robot\s*\??:?\*{0,2}\s*\??(.*?)\s*\*{0,2}\s*Objects\s*Involved\s*:\??\*{0,2}', 
                re.DOTALL | re.IGNORECASE
            )
            
            if not pattern.search(subtask):
                if llm and gpt_version:
                    print("Fixing structure of subtask with LLM")
                    fix_prompt = (
                        "The following subtask description needs to be reformatted. Please reformat it to strictly follow this structure:\n\n"
                        "#SubTask [number]: [Task Name]\n\n"
                        "# Initial Precondition analyze due to previous subtask:\n"
                        "#1. [precondition description]\n\n"
                        "[Action descriptions with Parameters, Preconditions, and Effects]\n\n"
                        "**Assigned Robot**: [robot number or 'team']\n"
                        "**Objects Involved**: [list of objects]\n\n"
                        "Important formatting rules:\n"
                        "1. Each section must start with the exact headers shown above\n"
                        "2. The order must be: SubTask header, Preconditions, Action descriptions, Assigned Robot, Objects Involved\n"
                        "3. Use '**Assigned Robot**:' and '**Objects Involved**:' exactly as shown with double asterisks\n"
                        "4. Include all action descriptions with their Parameters, Preconditions, and Effects\n"
                        "5. Keep the original action descriptions if they exist\n\n"
                        "Original subtask:\n" + subtask + "\n\n"
                        "Please provide ONLY the reformatted version following the structure above. Do not add any explanations or additional text."
                    )
                    
                    if "gpt" not in gpt_version:
                        _, fixed_subtask = llm.query_model(fix_prompt, gpt_version, max_tokens=1000, stop=["def"], frequency_penalty=0.30)
                    else:
                        messages = [
                            {"role": "system", "content": "You are a Robot PDDL problem Expert. Your task is to reformat subtask descriptions to match a specific structure. Do not add any explanations or additional text."},
                            {"role": "user", "content": fix_prompt}
                        ]
                        _, fixed_subtask = llm.query_model(messages, gpt_version, max_tokens=1400, frequency_penalty=0.4)
                    
                    # Verify the fix worked
                    if pattern.search(fixed_subtask):
                        fixed_subtasks.append(fixed_subtask)
                    else:
                        print("LLM structure fix failed, using original subtask")
                        fixed_subtasks.append(subtask)
                else:
                    print("LLM handler not provided, using original subtask")
                    fixed_subtasks.append(subtask)
            else:
                fixed_subtasks.append(subtask)
        
        return fixed_subtasks, sequence_operations

    def extract_domain_name(self, problem_file_path: str) -> Optional[str]:
        """Extract the domain name from a problem PDDL file.
        
        Args:
            problem_file_path (str): Path to the problem PDDL file
            
        Returns:
            Optional[str]: Domain name if found, None otherwise
        """
        try:
            domain_pattern = re.compile(r'\(\s*:domain\s+(\S+)\s*\)')
            content = self.read_file(problem_file_path)
            match = domain_pattern.search(content)
            return match.group(1) if match else None
        except Exception as e:
            print(f"Error extracting domain name from {problem_file_path}: {str(e)}")
            return None

    def find_domain_file(self, domain_name: str) -> Optional[str]:
        """Find the domain file for a given domain name.
        
        Args:
            domain_name (str): Name of the domain to find
            
        Returns:
            Optional[str]: Path to domain file if found, None otherwise
        """
        try:
            domain_path = os.path.join(self.base_path, "resources", f"{domain_name}.pddl")
            return domain_path if os.path.isfile(domain_path) else None
        except Exception as e:
            print(f"Error finding domain file for {domain_name}: {str(e)}")
            return None

    def clean_directory(self, directory_path: str) -> None:
        """Clean a directory by removing all files and subdirectories.
        
        Args:
            directory_path (str): Path to directory to clean
        """
        if os.path.exists(directory_path):
            for filename in os.listdir(directory_path):
                file_path = os.path.join(directory_path, filename)
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)

    def extract_plan_from_output(self, content: str) -> str:
        """Extract clean plan from planner output.
        
        Args:
            content (str): Raw planner output content
            
        Returns:
            str: Cleaned plan text
        """
        if not content or not isinstance(content, str):
            raise ValueError("Invalid content provided to extract_plan_from_output")
            
        try:
            plan_pattern = re.compile(r"^\s*\w+\s+\w+\s+\w+\s+\(\d+\)\s*$", re.MULTILINE)
            plan = plan_pattern.findall(content)
            return "\n".join(plan) if plan else ""
        except Exception as e:
            print(f"Error extracting plan from output: {str(e)}")
            return ""

    def calculate_task_completion_rate(self) -> Tuple[int, int]:
        """Calculate task completion rate from plan files.
        
        Returns:
            Tuple[int, int]: (number of completed tasks, total number of tasks)
        """
        TC = 0
        total_subtasks = 0

        for file_path in glob.glob(os.path.join(self.subtask_path, '*_plan.txt')):
            total_subtasks += 1
            content = self.read_file(file_path)
            TC += content.count('Solution found!')
        
        return TC, total_subtasks

    def parse_bddl_file(self, bddl_file_path: str) -> Dict[str, Any]:
        """Parse BDDL file and extract key components.
        
        Args:
            bddl_file_path (str): Path to BDDL file
            
        Returns:
            Dict with keys:
                - task_name: str
                - objects: List[Dict]
                - init_state: List[str]
                - goal_state: List[str]
        """
        try:
            content = self.read_file(bddl_file_path)
            
            # Extract task name from problem definition
            task_pattern = r'\(define \(problem (.*?)\)'
            task_match = re.search(task_pattern, content)
            task_name = task_match.group(1) if task_match else ""
            
            # Extract objects section
            objects_pattern = r'\(:objects(.*?)\)'
            objects_match = re.search(objects_pattern, content, re.DOTALL)
            objects_section = objects_match.group(1) if objects_match else ""
            
            # Extract init state
            init_pattern = r'\(:init(.*?)\)'
            init_match = re.search(init_pattern, content, re.DOTALL)
            init_state = init_match.group(1) if init_match else ""
            
            # Extract goal state
            goal_pattern = r'\(:goal(.*?)\)\s*\)'
            goal_match = re.search(goal_pattern, content, re.DOTALL)
            goal_state = goal_match.group(1) if goal_match else ""
            
            return {
                "task_name": task_name,
                "objects": objects_section.strip(),
                "init_state": init_state.strip(),
                "goal_state": goal_state.strip()
            }
        except Exception as e:
            raise PDDLError(f"Error parsing BDDL file: {str(e)}")

class LLMHandler:
    """Handles interactions with Language Models (LLMs).
    

    """
    
    def __init__(self, api_key_file: str):
        """Initialize the LLM handler.
        
        Args:
            api_key_file (str): Path to the API key file
        """
        self.setup_api(api_key_file)
    
    def setup_api(self, api_key_file: str) -> None:
        """Set up the OpenAI API key."""
        try:
            
            try:
                api_key = Path(api_key_file + '.txt').read_text().strip()
                if not api_key:
                    raise ValueError("API key file is empty")
                openai.api_key = api_key
                print("Successfully loaded API key from", api_key_file + '.txt')
            except FileNotFoundError:
                # Try without .txt extension
                try:
                    api_key = Path(api_key_file).read_text().strip()
                    if not api_key:
                        raise ValueError("API key file is empty")
                    openai.api_key = api_key
                    print("Successfully loaded API key from", api_key_file)
                except FileNotFoundError:
                    raise LLMError(f"API key file not found: {api_key_file} or {api_key_file}.txt")
        except Exception as e:
            raise LLMError(f"Error reading API key file: {str(e)}")
    
    def query_model(
        self, 
        prompt: Union[str, List[Dict]], 
        gpt_version: str, 
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        stop: Optional[List[str]] = None,
        logprobs: Optional[int] = 1,
        frequency_penalty: float = 0
    ) -> Tuple[dict, str]:
        """the language model 
        
        Args:
            prompt: Either a string or a list of message dicts
            gpt_version: The model version to use
            max_tokens: Maximum number of tokens in the response
            temperature: Sampling temperature
            stop: Optional list of stop sequences
            logprobs: Optional number of logprobs to return
            frequency_penalty: Frequency penalty for token generation
        
        Returns:
            Tuple of (full response object, generated text)
            
        """
        retry_delay = DEFAULT_RETRY_DELAY
        
        for attempt in range(MAX_RETRIES):
            try:
                if "gpt" not in gpt_version:
                    response = openai.completions.create(
                        model=gpt_version, 
                        prompt=prompt, 
                        max_tokens=max_tokens, 
                        temperature=temperature, 
                        stop=stop, 
                        logprobs=logprobs, 
                        frequency_penalty=frequency_penalty
                    )
                    return response, response.choices[0].text.strip()
                else:
                    response = openai.chat.completions.create(
                        model=gpt_version, 
                        messages=prompt, 
                        max_tokens=max_tokens, 
                        temperature=temperature, 
                        frequency_penalty=frequency_penalty
                    )
                    return response, response.choices[0].message.content.strip()
                    
            except openai.RateLimitError:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                raise LLMError("Rate limit exceeded")
                
            except (openai.APIError, openai.APITimeoutError) as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(retry_delay)
                    continue
                raise LLMError(f"API Error after all retries: {str(e)}")
                
            except Exception as e:
                raise LLMError(f"Unexpected error in LLM query: {str(e)}")

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
                _, validated_text = self.llm.query_model(messages, self.gpt_version, max_tokens=1400, frequency_penalty=0.4)
            
            # Save the validated content back to the problem file
            self.file_processor.write_file(problem_file, validated_text)
            
        except Exception as e:
            raise ValidationError(f"Error validating PDDL problem: {str(e)}")

class PDDLPlanner:
    
    def __init__(self, base_path: str, file_processor: FileProcessor):
        """Initialize the PDDL planner.

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
                text=True,
                timeout=300  # 5 minute timeout
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
    """Manages task processing and coordination.
    
    This is the main orchestrator class that coordinates all operations
 result logging.
    """
    
    def __init__(self, base_path: str, gpt_version: str, api_key_file: str, prompt_decompse_set: str = "pddl_train_task_decomposesep", prompt_allocation_set: str = "pddl_train_task_allocationsep"):
        """Initialize the task manager.
        
        Args:
            base_path (str): Base path for all operations
            gpt_version (str): Version of GPT to use
            api_key_file (str): Path to the API key file
            prompt_decompse_set (str): Name of the decomposition prompt set
            prompt_allocation_set (str): Name of the allocation prompt set
        """
        self.base_path = base_path
        self.gpt_version = gpt_version
        self.prompt_decompse_set = prompt_decompse_set
        self.prompt_allocation_set = prompt_allocation_set
        
        # Initialize components
        self.llm = LLMHandler(api_key_file)
        self.file_processor = FileProcessor(base_path)
        self.validator = PDDLValidator(self.llm, self.file_processor)
        self.planner = PDDLPlanner(base_path, self.file_processor)
        
        # Initialize paths
        self.resources_path = os.path.join(base_path, "resources")
        self.logs_path = os.path.join(".", "logs")  
        os.makedirs(self.logs_path, exist_ok=True)
        
        # Clean generated subtask directory
        self.clean_generated_subtask_directory()
        
        # Initialize result storage
        self.decomposed_plan: List[str] = []
        self.allocated_plan: List[str] = []
        self.code_plan: List[str] = []
        self.combined_plan: List[str] = []
        self.code_planpddl: List[str] = []
        self.sequence_operations: str = ""  # Initialize sequence_operations
        
        # Get action mapping from actions module
        
        # Initialize objects_ai as None, will be set in process_tasks
        self.objects_ai = None

    def clean_generated_subtask_directory(self):
        """Clean the generated subtask directory."""
        directory = os.path.join(self.resources_path, "generated_subtask")
        try:
            if os.path.exists(directory):
                for filename in os.listdir(directory):
                    file_path = os.path.join(directory, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"Error cleaning {file_path}: {str(e)}")
        except Exception as e:
            print(f"Error accessing directory {directory}: {str(e)}")
    
    def load_dataset(self, test_file: str) -> Tuple[List[str], List[List[dict]], List[str], List[int], List[int]]:
        """Load dataset from JSON file.
        
        Args:
            test_file (str): Path to the test file
            
        """
        test_tasks = []
        robots_test_tasks = []
        gt_test_tasks = []
        trans_cnt_tasks = []
        min_trans_cnt_tasks = []
        
        try:
            with open(test_file, "r") as f:
                for line in f.readlines():
                    values = list(json.loads(line).values())
                    test_tasks.append(values[0])
                    robots_test_tasks.append(values[1])
                    gt_test_tasks.append(values[2])
                    trans_cnt_tasks.append(values[3])
                    min_trans_cnt_tasks.append(values[4])
            
            # Prepare robot configurations
            available_robots = []
            for robots_list in robots_test_tasks:
                task_robots = []
                for i, r_id in enumerate(robots_list):
                    rob = robots.robots[r_id-1]  # Direct reference like original
                    rob['name'] = f'robot{i+1}'  # Use f-string for consistency
                    task_robots.append(rob)
                available_robots.append(task_robots)
            
            return test_tasks, available_robots, gt_test_tasks, trans_cnt_tasks, min_trans_cnt_tasks
            
        except FileNotFoundError:
            raise PDDLError(f"Test file not found: {test_file}")
        except json.JSONDecodeError as e:
            raise PDDLError(f"Error parsing JSON in test file: {str(e)}")
        except Exception as e:
            raise PDDLError(f"Error loading dataset: {str(e)}")
    
    def log_results(self, task: str, idx: int, available_robots: List[dict], 
                   gt_test_tasks: List[str], trans_cnt_tasks: List[int], 
                   min_trans_cnt_tasks: List[int], objects_ai: str,
                   bddl_file_path: Optional[str] = None):
        """Log results including BDDL file if provided."""
        print(f"\n[DEBUG] Logging task {idx + 1}")
        print(f"Current list lengths:")
        print(f"- code_planpddl: {len(self.code_planpddl)}")
        print(f"- combined_plan: {len(self.combined_plan)}")
        print(f"- decomposed_plan: {len(self.decomposed_plan)}")
        print(f"- allocated_plan: {len(self.allocated_plan)}")
        print(f"- code_plan: {len(self.code_plan)}")
        
        date_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        task_name = "_".join(task.split()).replace('\n', '')
        folder_name = f"{task_name}_plans_{date_time}"
        log_folder = os.path.join(self.logs_path, folder_name)
        
        print(f"Creating log folder: {log_folder}")
        os.makedirs(log_folder)
        
        try:
            print(f"Writing plans for task {idx + 1}")
            self._write_plan(log_folder, "code_planpddl.py", self.code_planpddl[idx])
            print(f"Successfully wrote code_planpddl for task {idx + 1}")
            self._write_plan(log_folder, "combined_plan.py", self.combined_plan[idx])
            print(f"Successfully wrote combined_plan for task {idx + 1}")
            self._write_plan(log_folder, "decomposed_plan.py", self.decomposed_plan[idx])
            print(f"Successfully wrote decomposed_plan for task {idx + 1}")
            self._write_plan(log_folder, "allocated_plan.py", self.allocated_plan[idx])
            print(f"Successfully wrote allocated_plan for task {idx + 1}")
            self._write_plan(log_folder, "code_plan.py", self.code_plan[idx])
            print(f"Successfully wrote code_plan for task {idx + 1}")
            
            # Log main information
            TC, total_subtasks = self.planner.calculate_completion_rate()
            with open(os.path.join(log_folder, "log.txt"), 'w') as f:
                f.write(task)
                f.write(f"\n\nGPT Version: {self.gpt_version}")
                f.write(f"\n{objects_ai}")
                f.write(f"\nrobots = {available_robots[idx]}")
                f.write(f"\nground_truth = {gt_test_tasks[idx]}")
                f.write(f"\ntrans = {trans_cnt_tasks[idx]}")
                f.write(f"\nmin_trans = {min_trans_cnt_tasks[idx]}")
                f.write(f"\nTotalsuccesssubtask = {TC}")
                f.write(f"\nTotalsubtask = {total_subtasks}")
            
            # Copy generated subtasks
            subtask_folder = os.path.join(log_folder, "generated_subtask")
            os.makedirs(subtask_folder)
            source_folder = os.path.join(self.resources_path, "generated_subtask")
            for file_name in os.listdir(source_folder):
                full_file_name = os.path.join(source_folder, file_name)
                if os.path.isfile(full_file_name):
                    shutil.copy(full_file_name, subtask_folder)
            
            # Add BDDL file to logs if provided
            if bddl_file_path and os.path.exists(bddl_file_path):
                bddl_content = self.file_processor.read_file(bddl_file_path)
                bddl_output_path = os.path.join(log_folder, "task.bddl")
                self.file_processor.write_file(bddl_output_path, bddl_content)
            
        except Exception as e:
            print(f"Error writing plans for task {idx + 1}: {str(e)}")

    def _write_plan(self, folder: str, filename: str, content: Union[str, List]):
        """Write a plan to a file."""
        if isinstance(content, list):
            for i, item in enumerate(content):
                with open(os.path.join(folder, f"{filename}.{i}"), 'w') as f:
                    f.write(str(item))
        else:
            with open(os.path.join(folder, filename), 'w') as f:
                f.write(content)

    def process_tasks(self, test_tasks: List[str], available_robots: List[dict], objects_ai: str) -> None:
        """Process a list of tasks."""
        try:
            # Initial task count
            print(f"\n[DIAGNOSTIC] Initial Task Count: {len(test_tasks)}")
            
            # Store objects_ai for use in other methods
            self.objects_ai = objects_ai
            
            # Initialize or reset result lists
            self.decomposed_plan = []
            self.allocated_plan = []
            self.code_plan = []
            self.combined_plan = []
            self.code_planpddl = []
            
            # Get domain content
            allaction_domain_path = os.path.join(self.resources_path, "allactionrobot.pddl")
            domain_content = self.file_processor.read_file(allaction_domain_path)
            
            # Process each task
            for task_idx, (task, robots) in enumerate(zip(test_tasks, available_robots)):
                print(f"\n{'='*50}")
                print(f"Processing Task {task_idx + 1}/{len(test_tasks)}")
                print(f"{'='*50}")
                
                # Clean generated subtask directory before starting new task
                self.clean_generated_subtask_directory()
                
                # Generate and store decomposed plan
                decomposed_plan = self._generate_decomposed_plan(task, domain_content, robots, objects_ai)
                self.decomposed_plan.append(decomposed_plan)
                print("✓ Decomposed plan generated")
                
                # Generate and store allocation plan
                allocated_plan = self._generate_allocation_plan(decomposed_plan, robots, objects_ai)
                self.allocated_plan.append(allocated_plan)
                print("✓ Allocation plan generated")
                
                print("Waiting for content summary...")
                time.sleep(60)  
                
                # Generate problem summary
                problem_summary = self._generate_problem_summary(decomposed_plan, allocated_plan, robots)
                print("✓ Problem summary generated")
                
                print("Waiting to generate problem files...")
                time.sleep(60)  
                
                # Generate and store problem files
                code_plan = self._generate_problem_files(problem_summary)
                self.code_plan.append(code_plan)
                print("✓ Problem files generated")
                
                # Split into subtasks
                self.file_processor.split_pddl_tasks(code_plan)
                print("✓ Split into subtasks complete")
                
                print("Waiting for files to be processed...")
                time.sleep(50)
                
                # Validate and plan
                self._validate_and_plan()
                print("✓ Validation and planning complete")
                
                # Combine and process plans
                combined_plan = self._combine_all_plans(decomposed_plan)
                self.combined_plan.append(combined_plan)
                print("✓ Plans combined")
                
                # Match references and store final PDDL plan
                matched_plan = self._match_references_for_plan(combined_plan, objects_ai)
                self.code_planpddl.append(matched_plan)
                print("✓ References matched")
                
                # Calculate completion rate
                tc, total = self.planner.calculate_completion_rate()
                print(f"Task {task_idx + 1} completion rate: {tc}/{total}")
                
            print(f"\n{'='*50}")
            print(f"All {len(test_tasks)} tasks processed")
            print(f"{'='*50}")
            
        except Exception as e:
            print(f"\n[ERROR] Task Processing Failed:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print(f"Current task index: {task_idx if 'task_idx' in locals() else 'Not started'}")
            raise

    def _generate_decomposed_plan(self, task: str, domain_content: str, robots: List[dict], objects_ai: str) -> str:
        """Generate decomposed plan for a task."""
        try:
            # Read decomposition prompt file
            decompose_prompt_path = os.path.join(self.base_path, "data", "pythonic_plans", f"{self.prompt_decompse_set}.py")
            decompose_prompt = self.file_processor.read_file(decompose_prompt_path)
            
            # Construct the prompt incrementally like the original
            prompt = f"from pddl domain file with all possible actions: \n{domain_content}\n\n"
            prompt += objects_ai
            prompt += f"\nrobots = {robots}\n\n"
            prompt += "robot initiate 'as not inaction robot '(which defaults location too)\n\n"
            prompt += decompose_prompt
            prompt += "# GENERAL TASK DECOMPOSITION \n"
            prompt += "Decompose and parallel subtasks where ever possible.\n\n"
            prompt += f"# Task Description: {task}"
            
            if "gpt" not in self.gpt_version:
                _, text = self.llm.query_model(prompt, self.gpt_version, max_tokens=1000, stop=["def"], frequency_penalty=0.15)
            else:
                messages = [{"role": "user", "content": prompt}]
                _, text = self.llm.query_model(messages, self.gpt_version, max_tokens=1300, frequency_penalty=0.0)
            
            return text
            
        except Exception as e:
            raise PDDLError(f"Error generating decomposed plan: {str(e)}")
    
    def _generate_allocation_plan(self, decomposed_plan: str, robots: List[dict], objects_ai: str) -> str:
        """Generate allocation plan for decomposed tasks.
        
        """
        try:
            # Read allocation prompt file
            prompt_file = os.path.join(self.base_path, "data", "pythonic_plans", f"{self.prompt_allocation_set}_solution.py")
            with open(prompt_file, "r") as allocated_prompt_file:
                allocated_prompt = allocated_prompt_file.read()
            
            # Build prompt incrementally like the original
            prompt = "\n"
            prompt += allocated_prompt
            prompt += decomposed_plan
            prompt += f"\n# TASK ALLOCATION"
            prompt += f"\n# Scenario: There are {len(robots)} robots available. The task should be performed using the minimum number of robots necessary. Robot should be assigned to subtasks that match its skills and mass capacity. Using your reasoning come up with a solution to satisfy all constraints."
            prompt += f"\n\nrobots = {robots}"
            prompt += f"\n{objects_ai}"
            prompt += f"\n\n# IMPORTANT: The AI should ensure that the robots assigned to the tasks have all the necessary skills to perform the tasks. IMPORTANT: Determine whether the subtasks must be performed sequentially or in parallel, or a combination of both and allocate robots based on availability. "
            prompt += f"\n# SOLUTION\n"
            
            # Handle different GPT versions like the original
            if "gpt" not in self.gpt_version:
                # older versions of GPT
                _, text = self.llm.query_model(prompt, self.gpt_version, max_tokens=1000, stop=["def"], frequency_penalty=0.65)
            elif "gpt-3.5" in self.gpt_version:
                # gpt 3.5 and its variants
                messages = [{"role": "user", "content": prompt}]
                _, text = self.llm.query_model(messages, self.gpt_version, max_tokens=1500, frequency_penalty=0.35)
            else:          
                # gpt 4.0o
                messages = [{"role": "user", "content": prompt}]
                _, text = self.llm.query_model(messages, self.gpt_version, max_tokens=400, frequency_penalty=0.69)
            
            return text
            
        except Exception as e:
            raise PDDLError(f"Error generating allocation plan: {str(e)}")

    def _generate_problem_summary(self, decomposed_plans: Union[str, List[str]], allocated_plans: Union[str, List[str]], available_robots: Union[List[dict], List[List[dict]]]) -> List[str]:
        """Generate problem summaries from decomposed and allocated plans.
        

            
        Returns:
            List[str]: List of generated problem summaries
        """
        try:
            print("Generating Allocated summary...")
            
            # Convert single items to lists
            if isinstance(decomposed_plans, str):
                decomposed_plans = [decomposed_plans]
            if isinstance(allocated_plans, str):
                allocated_plans = [allocated_plans]
            if not isinstance(available_robots[0], list):
                available_robots = [available_robots]
            
            # Read summary prompt file
            prompt_file = os.path.join(self.base_path, "data", "pythonic_plans", f"{self.prompt_allocation_set}_summary.py")
            with open(prompt_file, "r") as code_prompt_file:
                code_prompt = code_prompt_file.read()
            
            # Build base prompt once
            base_prompt = " finish the problem content summary strictly following the example format"
            base_prompt += "\n\n" + code_prompt + "\n\n"
            
            code_plan = []
            # Process each plan
            for i, (plan, solution) in enumerate(zip(decomposed_plans, allocated_plans)):
                # Build prompt for this plan
                prompt = base_prompt + plan
                prompt += f"\n# TASK ALLOCATION"
                prompt += f"\n\nrobots = {available_robots[i]}"
                prompt += solution
                prompt += f"\n# problem content summary  \n"
                
                if "gpt" not in self.gpt_version:
                    # older versions of GPT
                    _, text = self.llm.query_model(prompt, self.gpt_version, max_tokens=1000, stop=["def"], frequency_penalty=0.30)
                else:            
                    # using variants of gpt 4 or 3.5
                    messages = [
                        {"role": "system", "content": "You are a Robot PDDL problem Expert"},
                        {"role": "user", "content": prompt}
                    ]
                    _, text = self.llm.query_model(messages, self.gpt_version, max_tokens=1400, frequency_penalty=0.4)
                
                code_plan.append(text)
            
            return code_plan
            
        except Exception as e:
            raise PDDLError(f"Error generating problem summary: {str(e)}")

    def _generate_problem_files(self, problem_summary: Union[str, List[str]]) -> List[str]:
        """Generate PDDL problem files from plans.
        
        """
        # Handle list input by taking first summary
        if isinstance(problem_summary, list):
            problem_summary = problem_summary[0]
        
        problem_pddl = []
        
        # Split into subtasks and sequence operations
        subtasks, sequence_operations = self.file_processor.split_and_store_tasks(
            problem_summary,
            llm=self.llm,
            gpt_version=self.gpt_version
        )
        
        # Store sequence operations for later use
        self.sequence_operations = sequence_operations
        
        # Process each subtask using the class method
        problem_pddl = self.problemextracting(
            subtasks=subtasks,
            llm=self.llm,
            gpt_version=self.gpt_version,
            file_processor=self.file_processor,
            objects_ai=self.objects_ai,
            prompt_allocation_set=self.prompt_allocation_set
        )
        
        return problem_pddl
    
    def problemextracting(
        self,
        subtasks: List[str],
        llm: LLMHandler,
        gpt_version: str,
        file_processor: FileProcessor,
        objects_ai: str,
        prompt_allocation_set: str
        ) -> List[str]:
        """Extract problem files from subtasks."""
        problem_pddl: List[str] = []
        
        for subtask in subtasks:
            print("subtasks are:")
            print(subtask)
            
            # Extract assigned robots using regex
            pattern = re.compile(r"\s*\*{0,2}\s*Assigned\s*Robot\s*\??:?\*{0,2}\s*\??(.*?)\s*\*{0,2}\s*Objects\s*Involved\s*:\??\*{0,2}", re.DOTALL | re.IGNORECASE)
            assigned_robots_match = pattern.search(subtask)
            print("assigned_robots_match is")
            print(assigned_robots_match)

            if not assigned_robots_match:
                print("Invalid subtask structure, skipping")
                continue

            assigned_robots = assigned_robots_match.group(1).strip()
            if "team" in assigned_robots.lower() or "allactionrobot" in assigned_robots.lower():
                print("this is a team task")
                # Handle team robots
                all_domain_contents = ""
                team_pattern = re.compile(r"\s*\*{0,2}\s*robot\s*\??\s*(\d+)\*{0,2}", re.IGNORECASE)
                robot_numbers = team_pattern.findall(assigned_robots)
                normalized_robot_numbers = [f"robot{num}" for num in robot_numbers]
                
                for robot in normalized_robot_numbers:
                    domain_path = os.path.join(self.base_path, "resources", f"{robot}.pddl")
                    domain_content = file_processor.read_file(domain_path)
                    all_domain_contents += domain_content

                problem_fileexamplepath = os.path.join(self.base_path, "data", "pythonic_plans", f"{prompt_allocation_set}_teamproblem.py")
                problem_examplecontent = file_processor.read_file(problem_fileexamplepath)

                prompt = (
                    "\n" + problem_examplecontent +
                    "Strictly follow the structure and finish the tasks like example\n"
                    "Subtask examination from action perspective:" + subtask +
                    "\nDomain file content:" + domain_content +
                    "\n based on the objects availiable below." + objects_ai +
                    "Task description: extract out the problem files, based on the objects above, "
                    "the precondition, actions and subtask examination.\n"
                    "#IMPORTANT, strictly follow the structure ,stop generate after the Problem file generation is done."
                )

                if "gpt" not in gpt_version:
                    _, text = llm.query_model(prompt, gpt_version, max_tokens=1000, stop=["def"], frequency_penalty=0.30)
                else:            
                    messages = [
                        {"role": "system", "content": "You are a Robot PDDL problem Expert"},
                        {"role": "user", "content": prompt}
                    ]
                    _, text = llm.query_model(messages, gpt_version, max_tokens=1400, frequency_penalty=0.4)

                problem_pddl.append(text)

            else:
                # Handle single robot
                robot_pattern = re.compile(r"\s*\*{0,2}\s*robot\s*\??\s*(\d+)\*{0,2}", re.IGNORECASE)
                robot_numbers = robot_pattern.findall(assigned_robots)
                normalized_robot_numbers = [f"robot{num}" for num in robot_numbers]
                
                if robot_numbers:
                    robotassignnumber = f"{normalized_robot_numbers[0].replace(' ', '')}.pddl"
                    domain_path = os.path.join(self.base_path, "resources", robotassignnumber)
                    print("this is a solo work")
                    print(domain_path)

                    domain_content = file_processor.read_file(domain_path)
                    problem_fileexamplepath = os.path.join(self.base_path, "data", "pythonic_plans", f"{prompt_allocation_set}_problem.py")
                    problem_examplecontent = file_processor.read_file(problem_fileexamplepath)

                    prompt = (
                        "\n" + problem_examplecontent +
                        " Finish the tasks like example\n"
                        "Subtask examination from action perspective:" + subtask +
                        "\nDomain file content:" + domain_content +
                        "\n based on the objects availiable for potential usage below." + objects_ai +
                        "Task description: generate the problem file. Based on the objects above, "
                        "the domain file precondition, actions and subtask examination. "
                        "IMPORTANT the robot initate strictly as not inaction and robot "
                        "(which includes location)\n"
                        "#IMPORTANT, strictly follow the structure ,stop generate after the Problem file generation is done."
                    )

                    if "gpt" not in gpt_version:
                        _, text = llm.query_model(prompt, gpt_version, max_tokens=1000, stop=["def"], frequency_penalty=0.30)
                    else:            
                        messages = [
                            {"role": "system", "content": "You are a Robot PDDL problem Expert"},
                            {"role": "user", "content": prompt}
                        ]
                        _, text = llm.query_model(messages, gpt_version, max_tokens=1400, frequency_penalty=0.4)

                    problem_pddl.append(text)
                    
        print("the problem_pddl is")
        print(problem_pddl)
        return problem_pddl
    
    def _validate_and_plan(self) -> None:
        """Validate and plan all problem files."""
        try:
            # First run LLM validator
            print("Running LLM validator...")
            self.run_llmvalidator()
            
            # Wait for validation to complete
            print("Waiting 50 seconds for validation to complete...")
            #time.sleep(50)
            
            # Then run planners
            print("Running planners...")
            self.run_planners()
            
        except Exception as e:
            raise PDDLError(f"Error in validation and planning: {str(e)}")

    def run_llmvalidator(self) -> None:
        """Run LLM validation on problem files."""
        try:
            problem_files = [f for f in os.listdir(self.file_processor.subtask_path) if f.endswith('.pddl')]
            
            for problem_file in problem_files:
                try:
                    problem_file_full = os.path.join(self.file_processor.subtask_path, problem_file)
                    domain_name = self.file_processor.extract_domain_name(problem_file_full)
                    if not domain_name:
                        print(f"No domain specified in {problem_file}")
                        continue

                    domain_file = self.file_processor.find_domain_file(domain_name)
                    if not domain_file:
                        print(f"No domain file found for domain {domain_name}")
                        continue

                    domain_content = self.file_processor.read_file(domain_file)
                    problem_content = self.file_processor.read_file(problem_file_full)

                    prompt = (f"Domain Description:\n"
                            f"{domain_content}\n\n"
                            f"Problem Description:\n"
                            f"{problem_content}\n\n"
                            "Validate the preconditions in problem file to ensure all precondition listed object "
                            "is included and also in domain file, and go over structure to check the parenthesis "
                            "and syntext. Check and return only the validated problem file.")

                    if "gpt" not in self.gpt_version:
                        _, text = self.llm.query_model(prompt, self.gpt_version, max_tokens=1000, stop=["def"], frequency_penalty=0.30)
                    else:
                        messages = [{"role": "system", "content": "You are a Robot PDDL problem Expert"},
                                {"role": "user", "content": prompt}]
                        _, text = self.llm.query_model(messages, self.gpt_version, max_tokens=1400, frequency_penalty=0.4)

                    code_plan = [text]
                    self.file_processor.split_pddl_tasks(code_plan)
                    
                except Exception as e:
                    print(f"Error processing file {problem_file}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error in run_llmvalidator: {str(e)}")
            raise

    def run_planners(self) -> None:
        """Run PDDL planners on problem files."""
        try:
            planner_path = os.path.join(self.base_path, "downward", "fast-downward.py")
            problem_files = [f for f in os.listdir(self.file_processor.subtask_path) if f.endswith('.pddl')]

            for problem_file in problem_files:
                try:
                    problem_file_full = os.path.join(self.file_processor.subtask_path, problem_file)
                    domain_name = self.file_processor.extract_domain_name(problem_file_full)
                    if not domain_name:
                        print(f"No domain specified in {problem_file}")
                        continue

                    domain_file = self.file_processor.find_domain_file(domain_name)
                    if not domain_file:
                        print(f"No domain file found for domain {domain_name}")
                        continue

                    command = [
                        planner_path,
                        "--alias",
                        "seq-opt-lmcut",
                        domain_file,
                        problem_file_full
                    ]

                    result = subprocess.run(
                        command,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=300  # 5 minute timeout
                    )
                    
                    output_file = os.path.join(self.file_processor.subtask_path, problem_file.replace('.pddl', '_plan.txt'))
                    self.file_processor.write_file(output_file, result.stdout)

                    if result.stderr:
                        print(f"Warnings/Errors for {problem_file}:", result.stderr)
                        
                except subprocess.TimeoutExpired:
                    print(f"Planner timed out for {problem_file}")
                except Exception as e:
                    print(f"Error processing file {problem_file}: {str(e)}")
                    continue
                    
        except Exception as e:
            print(f"Error in run_planners: {str(e)}")
            raise

    def _combine_all_plans(self, decomposed_plan: Union[str, List[str]]) -> str:
        """Combine all generated plan files into a single plan.
 
        """
        # Handle list input by taking first plan
        if isinstance(decomposed_plan, list):
            decomposed_plan = decomposed_plan[0]
        
        base_path = os.path.join(self.resources_path, "generated_subtask")
        plan_files = [f for f in os.listdir(base_path) if f.endswith('_plan.txt')]
        
        prompt = ""
        # Add plans from files if they exist
        if plan_files:
            for idx, filename in enumerate(plan_files):
                filepath = os.path.join(base_path, filename)
                content = self.file_processor.read_file(filepath)
                plan = self.file_processor.extract_plan_from_output(content)
                if plan:  # Only add non-empty plans
                    prompt += f"\nPlan {idx + 1}:\n{plan}\n"
        
        # Add allocation examination and initial plan
        prompt += "\nallocation examination\n"
        prompt += self.sequence_operations if hasattr(self, 'sequence_operations') else ""
        prompt += "\ninitial plan examination\n"
        prompt += decomposed_plan
        
        prompt += ("\nyou are robot allocation expert, Your task is, based on inital plan examination "
                  "and allocation examination correct the subplans. Then based on your understanding "
                  "merge the subtasks together by using timed durative actions format, where parallel "
                  "tasks are performed at the same time. IMPORTANT: all 'variablelocation' should be "
                  "corrected to variable itself, since variable itself includes location. and result "
                  "must be in PDDL plan format.")
        
        if "gpt" not in self.gpt_version:
            _, text = self.llm.query_model(prompt, self.gpt_version, max_tokens=1000, stop=["def"], frequency_penalty=0.15)
        else:
            messages = [{"role": "user", "content": prompt}]
            _, text = self.llm.query_model(messages, self.gpt_version, max_tokens=1300, frequency_penalty=0.0)
        
        return text

    def _match_references_for_plan(self, plan: str, objects_ai: str) -> str:
        """Match and correct variable locations in plan."""
        prompt = (
            f"{objects_ai}\n"
            "IMPORTANT: Your TASK is based on the provided pddl plan provided in the passage below "
            "and the object list above, modify and only modify the plan so that all 'variablelocation' "
            "should be corrected to variable itself, since variable itself includes location. and similarly variable names should be corrected to variable itself. "
            "IMPORTANT: the only parenthesis usage should be for the correct PDDL plan, no exception.\n\n"
            f"{plan}"
        )
        
        if "gpt" not in self.gpt_version:
            _, text = self.llm.query_model(prompt, self.gpt_version, max_tokens=1000, stop=["def"], frequency_penalty=0.15)
        else:
            messages = [{"role": "user", "content": prompt}]
            _, text = self.llm.query_model(messages, self.gpt_version, max_tokens=1300, frequency_penalty=0.0)
        
        return text

    def process_bddl_task(self, bddl_file_path: str, available_robots: List[dict]) -> None:
        """Process a task from BDDL file format.
        
        Args:
            bddl_file_path (str): Path to BDDL file
            available_robots (List[dict]): List of available robots
        """
        # Parse BDDL file
        bddl_data = self.file_processor.parse_bddl_file(bddl_file_path)
        
        # Convert task name to instruction
        task_instruction = bddl_data["task_name"].replace("-", " ").replace("_", " ")
        
        # Process task as before but with additional BDDL context
        self.process_tasks(
            test_tasks=[task_instruction],
            available_robots=[available_robots],
            objects_ai=bddl_data["objects"],
            bddl_context=bddl_data  # Pass full BDDL data for reference
        )

    def create_bddl_dataset(self, tasks: List[str], output_dir: str) -> None:
        """Create BDDL format files for a list of tasks.
        
        Args:
            tasks (List[str]): List of task descriptions
            output_dir (str): Directory to save BDDL files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for i, task in enumerate(tasks):
            # Generate BDDL content
            bddl_content = self._generate_bddl_content(
                task_name=task.lower().replace(" ", "_"),
                task_index=i,
                objects=self.objects_ai,  # Use existing objects
            )
            
            # Save BDDL file
            output_path = os.path.join(output_dir, f"problem{i}.bddl")
            self.file_processor.write_file(output_path, bddl_content)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl-file", type=str, help="Path to BDDL file")
    parser.add_argument(
        "--floor-plan", 
        type=int, 
        required=False,  # Changed from True
        help="Required unless --bddl-file is provided"
    )
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
    
    args = parser.parse_args()
    
    # Validate that either bddl_file or floor_plan is provided
    if not args.bddl_file and args.floor_plan is None:
        parser.error("Either --bddl-file or --floor-plan must be provided")
        
    return args

def main():
    """Main execution function."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Initialize task manager
        task_manager = TaskManager(
            base_path=os.getcwd(),
            gpt_version=args.gpt_version,
            api_key_file=args.openai_api_key_file,
            prompt_decompse_set=args.prompt_decompse_set,
            prompt_allocation_set=args.prompt_allocation_set
        )
        
        if args.bddl_file:
            # Process single BDDL task
            bddl_data = task_manager.file_processor.parse_bddl_file(args.bddl_file)
            print("\nBDDL Data:")
            print(f"Task Name: {bddl_data['task_name']}")
            print(f"Objects: {bddl_data['objects']}")
            print(f"Init State: {bddl_data['init_state']}")
            print(f"Goal State: {bddl_data['goal_state']}\n")
            
            # Convert task name to instruction
            task_instruction = bddl_data["task_name"].replace("-", " ").replace("_", " ")
            
            # Use a default robot configuration with more capabilities
            available_robots = [{
                "name": "robot1",
                "skills": ["grasp", "place", "pour", "move", "pick", "hold"],
                "mass_capacity": 10.0
            }]
            
            # Format objects for processing
            objects_ai = f"\n\nobjects = {bddl_data['objects']}"
            
            # Process the task
            task_manager.process_tasks(
                test_tasks=[task_instruction],
                available_robots=[available_robots],
                objects_ai=objects_ai
            )
            
            # Log results for BDDL task
            if args.log_results:
                task_manager.log_results(
                    task=task_instruction,
                    idx=0,
                    available_robots=available_robots,
                    gt_test_tasks=[""],  # No ground truth for BDDL tasks
                    trans_cnt_tasks=[0],
                    min_trans_cnt_tasks=[0],
                    objects_ai=objects_ai,
                    bddl_file_path=args.bddl_file
                )
        else:
            # Original workflow
            test_file = os.path.join("data", args.test_set, f"FloorPlan{args.floor_plan}.json")
            test_tasks, available_robots, gt_test_tasks, trans_cnt_tasks, min_trans_cnt_tasks = \
                task_manager.load_dataset(test_file)
            
            print(f"\n----Test set tasks----\n{test_tasks}\nTotal: {len(test_tasks)} tasks\n")
            
            # Get AI2thor objects 
            objects_ai = f"\n\nobjects = {PDDLUtils.get_ai2_thor_objects(args.floor_plan)}"
            
            # Process tasks with objects_ai
            task_manager.process_tasks(test_tasks, available_robots, objects_ai)
            
            # Log results if enabled
            if args.log_results:
                for idx, task in enumerate(test_tasks):
                    task_manager.log_results(
                        task=task,
                        idx=idx,
                        available_robots=available_robots,
                        gt_test_tasks=gt_test_tasks,
                        trans_cnt_tasks=trans_cnt_tasks,
                        min_trans_cnt_tasks=min_trans_cnt_tasks,
                        objects_ai=objects_ai
                    )
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        print(f"Full error: {str(e.__class__.__name__)}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()





























#####



    




