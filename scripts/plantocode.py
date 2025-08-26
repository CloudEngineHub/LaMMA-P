
import json
import argparse
import os
import re
import sys
import glob
from pathlib import Path
from datetime import datetime
import time
from typing import Dict, List, Any, Optional, Union, Tuple

import openai

# Constants
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.1
DEFAULT_RETRY_DELAY = 20
MAX_RETRIES = 3

class LLMError(Exception):
    """Exception raised for Language Model related errors."""
    pass

class MimicTranslationError(Exception):
    """Exception raised for Mimic translation errors."""
    pass

class LLMHandler:
    """Handles interactions with Language Models (LLMs) using OpenAI API."""
    
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
        """Query the language model using OpenAI API.
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

class MimicFormatTranslator:
    """Translates PDDL action sequences to mimic format using OpenAI API."""
    
    def __init__(self, api_key_file: str, gpt_version: str = "gpt-4o"):
        """Initialize the translator with OpenAI API.
        
        Args:
            api_key_file (str): Path to the API key file
            gpt_version (str): GPT model version to use
        """
        self.gpt_version = gpt_version
        self.llm = LLMHandler(api_key_file)
        print(f"Initialized MimicFormatTranslator with {gpt_version}")
    
    def create_few_shot_prompt(self, task_description: str, action_sequence: str) -> Union[str, List[Dict]]:
        """Create a few-shot prompt to translate PDDL actions to mimic format."""
        
        # Few-shot examples based on the AI2-THOR controller format
        few_shot_examples = """# Example 1: Washing an Apple
Task: Wash the apple
PDDL Actions: GoToObject, PickupObject, GoToObject, PutObject, SwitchOn, SwitchOff, PickupObject, GoToObject, PutObject

def wash_apple(robot):
    # 0: Task: Wash the Apple
    # 1: Go to the Apple.
    GoToObject(robot, 'Apple')
    # 2: Pick up the Apple.
    PickupObject(robot, 'Apple')
    # 3: Go to the Sink.
    GoToObject(robot, 'Sink')
    # 4: Put the Apple in the Sink.
    PutObject(robot, 'Apple', 'Sink')
    # 5: Switch on the Faucet.
    SwitchOn(robot, 'Faucet')
    # 6: Wait for a while to let the Apple wash.
    time.sleep(5)
    # 7: Switch off the Faucet.
    SwitchOff(robot, 'Faucet')
    # 8: Pick up the washed Apple.
    PickupObject(robot, 'Apple')
    # 9: Go to the CounterTop.
    GoToObject(robot, 'CounterTop')
    # 10: Put the washed Apple on the CounterTop.
    PutObject(robot, 'Apple', 'CounterTop')

# Example 2: Moving Objects
Task: Move the book from table to shelf
PDDL Actions: GoToObject, PickupObject, GoToObject, PutObject

def move_book(robot):
    # 0: Task: Move the book from table to shelf
    # 1: Go to the Book.
    GoToObject(robot, 'Book')
    # 2: Pick up the Book.
    PickupObject(robot, 'Book')
    # 3: Go to the Shelf.
    GoToObject(robot, 'Shelf')
    # 4: Put the Book on the Shelf.
    PutObject(robot, 'Book', 'Shelf')

# Example 3: Complex Task with Multiple Objects
Task: Prepare a snack by moving apple and bread to the counter
PDDL Actions: GoToObject, PickupObject, GoToObject, PutObject, GoToObject, PickupObject, GoToObject, PutObject

def prepare_snack(robot):
    # 0: Task: Prepare a snack by moving apple and bread to the counter
    # 1: Go to the Apple.
    GoToObject(robot, 'Apple')
    # 2: Pick up the Apple.
    PickupObject(robot, 'Apple')
    # 3: Go to the Counter.
    GoToObject(robot, 'Counter')
    # 4: Put the Apple on the Counter.
    PutObject(robot, 'Apple', 'Counter')
    # 5: Go to the Bread.
    GoToObject(robot, 'Bread')
    # 6: Pick up the Bread.
    PickupObject(robot, 'Bread')
    # 7: Go to the Counter.
    GoToObject(robot, 'Counter')
    # 8: Put the Bread on the Counter.
    PutObject(robot, 'Bread', 'Counter')

# Now translate the following task:
Task: {task_description}
PDDL Actions: {action_sequence}

def execute_task(robot):
    # 0: Task: {task_description}
"""
        
        # Return as string for older GPT models, or as messages for newer ones
        if "gpt" not in self.gpt_version:
            return few_shot_examples
        else:
            return [
                {"role": "system", "content": "You are a Robot PDDL to Mimic Format Translator. Your task is to translate PDDL action sequences into executable Python code following the AI2-THOR controller format. Follow the examples exactly."},
                {"role": "user", "content": few_shot_examples}
            ]
    
    def translate_to_mimic_format(self, task_description: str, action_sequence: str,
                                max_tokens: int = 1024,
                                temperature: float = 0.1,
                                frequency_penalty: float = 0.0) -> str:
        """Translate PDDL action sequence to mimic format using OpenAI API."""
        try:
            # Create few-shot prompt
            prompt = self.create_few_shot_prompt(task_description, action_sequence)
            
            # Query the model
            start_time = time.time()
            _, response = self.llm.query_model(
                prompt=prompt,
                gpt_version=self.gpt_version,
                max_tokens=max_tokens,
                temperature=temperature,
                frequency_penalty=frequency_penalty
            )
            translation_time = time.time() - start_time
            
            print(f"Translation completed in {translation_time:.2f}s")
            return response
            
        except Exception as e:
            raise MimicTranslationError(f"Error in mimic translation: {str(e)}")
    
    def extract_function_name(self, task_description: str) -> str:


        clean_task = re.sub(r'[^a-zA-Z0-9\s]', '', task_description.lower())
        words = clean_task.split()[:3]  # Take first 3 words
        function_name = '_'.join(words)
        return function_name

def load_pddl_results_from_logs(logs_dir: str) -> List[Dict[str, Any]]:
    """Load PDDL results from the log directories created by pddlrun_llmseparate.py."""
    
    results = []
    logs_path = Path(logs_dir)
    
    if not logs_path.exists():
        print(f"Logs directory not found: {logs_dir}")
        return results
    
    # Find all log folders (they end with _plans_YYYY-MM-DD-HH-MM-SS)
    log_folders = list(logs_path.glob("*_plans_*"))
    
    if not log_folders:
        print(f"No log folders found in {logs_dir}")
        return results
    
    print(f"Found {len(log_folders)} log folders")
    
    for folder in log_folders:
        try:
            # Read log.txt to get task information
            log_file = folder / "log.txt"
            if not log_file.exists():
                print(f"Warning: log.txt not found in {folder}")
                continue
            
            with open(log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            # Extract task description from log content
            lines = log_content.split('\n')
            task_description = lines[0] if lines else "Unknown task"
            
            # Read the final PDDL plan (code_planpddl.py)
            pddl_file = folder / "code_planpddl.py"
            if not pddl_file.exists():
                print(f"Warning: code_planpddl.py not found in {folder}")
                continue
            
            with open(pddl_file, 'r', encoding='utf-8') as f:
                pddl_content = f.read()
            
            # Extract action sequence from PDDL content
            action_sequence = extract_actions_from_pddl(pddl_content)
            
            # Create result entry
            result = {
                'episode_id': folder.name,
                'scene_id': 'pddl_generated',
                'task_description': task_description,
                'extracted_action_sequence': action_sequence,
                'pddl_content': pddl_content,
                'log_folder': str(folder)
            }
            
            results.append(result)
            print(f"  ✓ Loaded: {folder.name} - {task_description[:50]}...")
            
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")
            continue
    
    print(f"Successfully loaded {len(results)} PDDL results")
    return results

def extract_actions_from_pddl(pddl_content: str) -> str:
    """Extract action sequence from PDDL plan content."""
    
    # Look for action patterns in PDDL content
    action_patterns = [
        r'\(([a-zA-Z][a-zA-Z0-9_-]*)\s+[^)]*\)',  # (action_name ...)
        r'\(:action\s+([a-zA-Z][a-zA-Z0-9_-]*)',   # (:action action_name
        r'([a-zA-Z][a-zA-Z0-9_-]*)\s*\([^)]*\)',  # action_name(...)
    ]
    
    actions = []
    
    for pattern in action_patterns:
        matches = re.findall(pattern, pddl_content, re.IGNORECASE)
        for match in matches:
            action_name = match.strip()
            # Filter out common PDDL keywords
            if action_name.lower() not in ['define', 'domain', 'problem', 'requirements', 'types', 'predicates', 'action', 'parameters', 'precondition', 'effect', 'goal', 'init', 'objects']:
                actions.append(action_name)
    
    # Remove duplicates while preserving order
    unique_actions = []
    for action in actions:
        if action not in unique_actions:
            unique_actions.append(action)
    
    return ', '.join(unique_actions)

def load_extraction_results(results_file: str) -> List[Dict[str, Any]]:
    """Load results from JSON file (for backward compatibility)."""
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Loaded {len(results)} results from {results_file}")
        return results
    except Exception as e:
        print(f"Error loading results file {results_file}: {e}")
        return []

def process_results_for_mimic(results: List[Dict[str, Any]], translator: MimicFormatTranslator,
                            output_dir: str, batch_size: int = 3) -> List[Dict[str, Any]]:
    """Process all results to translate to mimic format."""
    processed_results = []
    
    print(f"Processing {len(results)} results for mimic format translation...")
    print(f"Processing in batches of {batch_size}")
    
    # Process in batches to manage API rate limits
    for batch_start in range(0, len(results), batch_size):
        batch_end = min(batch_start + batch_size, len(results))
        batch_results = results[batch_start:batch_end]
        
        print(f"\nProcessing batch {batch_start//batch_size + 1}/{(len(results) + batch_size - 1)//batch_size}")
        print(f"Tasks {batch_start + 1}-{batch_end}")
        
        for i, result in enumerate(batch_results):
            global_index = batch_start + i
            episode_id = result.get('episode_id', f'task_{global_index}')
            scene_id = result.get('scene_id', 'unknown')
            task_description = result.get('task_description', '')
            action_sequence = result.get('extracted_action_sequence', '')
            
            print(f"\n[{global_index + 1}/{len(results)}] Processing Episode {episode_id}, Scene {scene_id}")
            
            try:
                # Skip if no action sequence was extracted
                if not action_sequence or action_sequence.strip() == "":
                    print(f"  ⚠ No action sequence found, skipping")
                    processed_result = {
                        'episode_id': episode_id,
                        'scene_id': scene_id,
                        'task_description': task_description,
                        'original_action_sequence': action_sequence,
                        'mimic_format_code': None,
                        'function_name': None,
                        'translation_time': 0,
                        'success': False,
                        'error': 'No action sequence to translate'
                    }
                    processed_results.append(processed_result)
                    continue
                
                # Translate to mimic format using OpenAI API
                start_time = time.time()
                mimic_code = translator.translate_to_mimic_format(task_description, action_sequence)
                translation_time = time.time() - start_time
                
                # Extract function name
                function_name = translator.extract_function_name(task_description)
                
                # Create processed result
                processed_result = {
                    'episode_id': episode_id,
                    'scene_id': scene_id,
                    'task_description': task_description,
                    'original_action_sequence': action_sequence,
                    'mimic_format_code': mimic_code,
                    'function_name': function_name,
                    'translation_time': translation_time,
                    'extraction_time': result.get('extraction_time', 0),
                    'generation_time': result.get('generation_time', 0),
                    'success': len(mimic_code.strip()) > 0
                }
                
                processed_results.append(processed_result)
                
                print(f"  ✓ Translated to mimic format ({translation_time:.2f}s)")
                print(f"  Function name: {function_name}")
                print(f"  Code preview: {mimic_code[:100]}{'...' if len(mimic_code) > 100 else ''}")
                
            except Exception as e:
                print(f"  ✗ Error in mimic translation: {e}")
                processed_result = {
                    'episode_id': episode_id,
                    'scene_id': scene_id,
                    'task_description': task_description,
                    'original_action_sequence': action_sequence,
                    'error': str(e),
                    'success': False
                }
                processed_results.append(processed_result)
        
        # Add delay between batches to respect API rate limits
        if batch_start + batch_size < len(results):
            print(f"  Waiting 2 seconds before next batch...")
            time.sleep(2)
    
    return processed_results

def save_individual_mimic_files(processed_results: List[Dict[str, Any]], output_dir: str):
    """Save individual mimic format files for each successful translation."""
    mimic_dir = Path(output_dir) / "mimic_code_files"
    mimic_dir.mkdir(parents=True, exist_ok=True)
    
    successful_translations = [r for r in processed_results if r.get('success', False)]
    
    print(f"\nSaving {len(successful_translations)} individual mimic code files...")
    
    for i, result in enumerate(successful_translations):
        episode_id = result.get('episode_id', f'task_{i}')
        scene_id = result.get('scene_id', 'unknown')
        function_name = result.get('function_name', f'task_{i}')
        mimic_code = result.get('mimic_format_code', '')
        task_description = result.get('task_description', '')
        
        # Create filename
        filename = f"mimic_{episode_id}_{scene_id}_{function_name}.py"
        filepath = mimic_dir / filename
        
        # Create complete Python file content
        file_content = f"""#!/usr/bin/env python3
\"\"\"
Mimic Format Code for AI2-THOR Controller
Generated from PDDL Action Sequence Translation

Episode ID: {episode_id}
Scene ID: {scene_id}
Task: {task_description}
\"\"\"

import time
import threading

# Import AI2-THOR controller functions
# from ai2thor_controller import GoToObject, PickupObject, PutObject, SwitchOn, SwitchOff

{mimic_code}

# Example usage:
# robot = get_robot_instance()
# execute_task(robot)
"""
        
        # Save file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(file_content)
        
        print(f"  ✓ Saved: {filename}")
    
    print(f"All mimic code files saved to: {mimic_dir}")

def generate_summary(processed_results: List[Dict[str, Any]], output_dir: str):
    """Generate summary statistics and reports."""
    total_results = len(processed_results)
    successful_translations = sum(1 for r in processed_results if r.get('success', False))
    
    # Time statistics
    translation_times = [r.get('translation_time', 0) for r in processed_results if r.get('translation_time')]
    avg_translation_time = sum(translation_times) / len(translation_times) if translation_times else 0
    
    # Generate summary report
    summary = {
        'total_results': total_results,
        'successful_translations': successful_translations,
        'success_rate': successful_translations / total_results * 100 if total_results > 0 else 0,
        'average_translation_time': avg_translation_time,
        'total_translation_time': sum(translation_times)
    }
    
    # Save summary
    summary_file = Path(output_dir) / "mimic_translation_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Save detailed results
    results_file = Path(output_dir) / "mimic_translation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(processed_results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n=== MIMIC TRANSLATION SUMMARY ===")
    print(f"Total results processed: {total_results}")
    print(f"Successful translations: {successful_translations} ({summary['success_rate']:.1f}%)")
    print(f"Average translation time: {summary['average_translation_time']:.2f}s")
    print(f"Total translation time: {summary['total_translation_time']:.2f}s")
    
    print(f"\nResults saved to: {output_dir}")
    print(f"Summary: {summary_file}")
    print(f"Detailed results: {results_file}")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Translate PDDL action sequences to mimic format using OpenAI API. Can load from JSON files or PDDL log directories created by pddlrun_llmseparate.py')
    parser.add_argument('--openai-api-key-file', type=str, default="api_key",
                       help='Path to OpenAI API key file')
    parser.add_argument('--gpt-version', type=str, default="gpt-4o",
                       choices=['gpt-3.5-turbo', 'gpt-4o', 'gpt-3.5-turbo-16k'],
                       help='GPT model version to use')
    parser.add_argument('--input-source', type=str, choices=['json', 'pddl_logs'], default='pddl_logs',
                       help='Input source type: json file or pddl_logs directory')
    parser.add_argument('--input-file', type=str, 
                       default='../model_testing/70b_extracted_actions/70b_extracted_action_sequences.json',
                       help='Path to the extracted action sequences JSON file (for json input source)')
    parser.add_argument('--logs-dir', type=str, 
                       default='./logs',
                       help='Path to logs directory from pddlrun_llmseparate.py (for pddl_logs input source)')
    parser.add_argument('--output-dir', type=str, 
                       default='./mimic_translation_results',
                       help='Directory to save mimic translation results')
    parser.add_argument('--batch-size', type=int, default=3,
                       help='Number of tasks to process in each batch (default: 3)')
    parser.add_argument('--max-tokens', type=int, default=1024,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Sampling temperature')
    parser.add_argument('--frequency-penalty', type=float, default=0.0,
                       help='Frequency penalty for token generation')
    
    args = parser.parse_args()
    
    # Validate input based on source type
    if args.input_source == 'json' and not args.input_file:
        parser.error("--input-file must be provided for json input source")
    elif args.input_source == 'pddl_logs' and not args.logs_dir:
        parser.error("--logs-dir must be provided for pddl_logs input source")
        
    return args

def main():
    """Main execution function."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Load results based on input source
        if args.input_source == 'json':
            print(f"Loading extraction results from JSON file: {args.input_file}")
            results = load_extraction_results(args.input_file)
        else:  # pddl_logs
            print(f"Loading PDDL results from logs directory: {args.logs_dir}")
            results = load_pddl_results_from_logs(args.logs_dir)
        
        if not results:
            print("No results found!")
            return
        
        # Create output directory
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize translator with OpenAI API
        translator = MimicFormatTranslator(
            api_key_file=args.openai_api_key_file,
            gpt_version=args.gpt_version
        )
        
        # Process results for mimic translation
        processed_results = process_results_for_mimic(
            results=results,
            translator=translator,
            output_dir=args.output_dir,
            batch_size=args.batch_size
        )
        
        # Save individual mimic files
        save_individual_mimic_files(processed_results, args.output_dir)
        
        # Generate summary
        generate_summary(processed_results, args.output_dir)
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        print(f"Full error: {str(e.__class__.__name__)}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 