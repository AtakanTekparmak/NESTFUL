from typing import List, Dict, Any, Optional, Tuple
import json
import logging
from tqdm import tqdm
import argparse

from evaluation.model import get_completion, get_completions_batch
from evaluation.utils import (
    load_system_prompt,
    load_data,
    load_function_map,
    resolve_function,
)
from evaluation.schemas import NestfulRow, ToolCall

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NESTFULEvaluator:
    def __init__(
        self,
        model_name: str,
        provider: str,
        debug: bool = True
    ):
        """Initialize the evaluator with model and settings"""
        self.model_name = model_name
        self.provider = provider
        self.debug = debug
        self.function_map = load_function_map()
        logger.info(f"Initialized evaluator with model: {model_name}")

    def parse_function_sequence(self, model_output: str) -> Optional[List[Dict]]:
        """Parse the function sequence from the model output"""
        try:
            # Try to find JSON list in code blocks
            json_blocks = []
            lines = model_output.split('\n')
            in_code_block = False
            current_block = []
            
            for line in lines:
                if line.startswith('```'):
                    if in_code_block:
                        json_blocks.append('\n'.join(current_block))
                        current_block = []
                    in_code_block = not in_code_block
                    continue
                if in_code_block:
                    current_block.append(line)
            
            # Try each JSON block
            for block in json_blocks:
                try:
                    # Remove any "json" or other language identifiers
                    block = block.replace('json\n', '').strip()
                    sequence = json.loads(block)
                    
                    # Validate sequence format
                    if not isinstance(sequence, list):
                        continue
                        
                    # Convert named arguments to positional arguments
                    for call in sequence:
                        if not all(key in call for key in ['name', 'label', 'arguments']):
                            continue
                            
                        # Convert named arguments to positional arguments
                        args = call['arguments']
                        if not all(k.startswith('arg_') for k in args.keys()):
                            # Convert named args to positional args
                            new_args = {}
                            for i, (_, value) in enumerate(sorted(args.items())):
                                new_args[f'arg_{i}'] = value
                            call['arguments'] = new_args
                    
                    logger.info(f"Parsed function sequence from json code block: {str(sequence)[:100]}...")
                    return sequence
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.warning(f"Error parsing json block: {str(e)}")
                    continue
            
            logger.warning("No valid JSON found in json code blocks")
            return None
            
        except Exception as e:
            logger.error(f"Error parsing function sequence: {str(e)}")
            return None

    def execute_function_sequence(
        self,
        sequence: List[Dict]
    ) -> Tuple[Optional[float], Dict]:
        """Execute a sequence of function calls and track variables"""
        variables = {}
        final_result = None
        
        for i, func_call in enumerate(sequence):
            try:
                # Get the function implementation
                func_name = func_call['name']
                if self.debug:
                    logger.info(f"Executing function {i+1}/{len(sequence)}: {func_name}")
                
                # Resolve the function
                func = resolve_function(func_name, self.function_map)
                
                # Prepare arguments
                args = []
                for arg_name in sorted(func_call['arguments'].keys()):  # Sort to ensure consistent order
                    arg_value = func_call['arguments'][arg_name]
                    if isinstance(arg_value, str) and arg_value.startswith('$') and arg_value.endswith('$'):
                        # Handle variable reference
                        var_ref = arg_value[1:-1]
                        var_name, field = var_ref.split('.')
                        if var_name not in variables:
                            raise ValueError(f"Unknown variable reference: {var_name}")
                        if field == 'result':
                            args.append(variables[var_name]['result'])
                        else:
                            args.append(variables[var_name][field])
                    else:
                        args.append(arg_value)
                
                if self.debug:
                    logger.info(f"Function arguments: {args}")
                
                # Execute function with unpacked arguments
                result = func(*args)  # Use positional arguments
                
                # Store result
                label = func_call['label']
                variables[label] = {'result': result}
                final_result = result
                
                if self.debug:
                    logger.info(f"Function result stored in {label}: {result}")
                
            except Exception as e:
                logger.error(f"Error executing {func_name}: {str(e)}")
                return None, variables
        
        return final_result, variables

    def _normalize_param(self, value: Any) -> str:
        """Normalize parameter values for comparison"""
        if isinstance(value, str):
            if value.startswith('$') and value.endswith('$'):
                # For variable references, just compare the variable name and field
                parts = value[1:-1].split('.')
                return f"var_{parts[0]}.{parts[1]}" if len(parts) > 1 else parts[0]
            return value.lower()  # Case-insensitive comparison for strings
        elif isinstance(value, (int, float)):
            # Convert numbers to string with limited precision
            return f"{float(value):.6f}"
        elif isinstance(value, list):
            # For lists, convert each element
            return '[' + ','.join(self._normalize_param(v) for v in value) + ']'
        return str(value)

    def _compare_params(self, pred_args: Dict, gold_args: Dict) -> float:
        """Compare parameters and return a similarity score"""
        if not pred_args or not gold_args:
            return 0.0
        
        # Get all unique keys
        all_keys = set(pred_args.keys()) | set(gold_args.keys())
        matches = 0
        total = 0
        
        for key in sorted(all_keys):  # Sort to ensure consistent order
            total += 1
            if key in pred_args and key in gold_args:
                pred_val = self._normalize_param(pred_args[key])
                gold_val = self._normalize_param(gold_args[key])
                
                # For variable references, compare just the variable part
                if isinstance(pred_args[key], str) and pred_args[key].startswith('$'):
                    pred_var = pred_val.split('.')[0]
                    gold_var = gold_val.split('.')[0]
                    if pred_var == gold_var:
                        matches += 0.5  # Partial match for variable name
                        if pred_val == gold_val:  # Full match including the field
                            matches += 0.5
                # For numeric values, allow small differences
                elif isinstance(pred_args[key], (int, float)) and isinstance(gold_args[key], (int, float)):
                    if abs(float(pred_args[key]) - float(gold_args[key])) < 1e-6:
                        matches += 1
                # For other types, exact match
                elif pred_val == gold_val:
                    matches += 1
        
        return matches / total if total > 0 else 0.0

    def calculate_metrics(
        self,
        predicted: List[Dict],
        gold: List[Dict],
        executed_result: Optional[Any],
        gold_answer: Any
    ) -> Dict:
        """Calculate all evaluation metrics"""
        metrics = {
            'f1_function': 0.0,
            'f1_param': 0.0,
            'partial_match': 0.0,
            'full_match': 0.0,
            'win_rate': 0.0
        }
        
        try:
            if not predicted or not gold:
                return metrics
                
            # Convert gold to list of dicts if it's a list of ToolCall objects
            if isinstance(gold, list) and len(gold) > 0 and hasattr(gold[0], '__dict__'):
                gold = [
                    {
                        'name': g.name,
                        'label': g.label,
                        'arguments': {f'arg_{i}': v for i, v in enumerate(g.arguments)}
                    }
                    for g in gold
                ]
            
            # F1 for function names - stricter matching
            pred_funcs = [(i, call['name']) for i, call in enumerate(predicted)]
            gold_funcs = [(i, call['name']) for i, call in enumerate(gold)]
            metrics['f1_function'] = self._calculate_f1(pred_funcs, gold_funcs)
            
            # F1 for parameters - compare each function's parameters separately
            param_scores = []
            for i, (p, g) in enumerate(zip(predicted, gold)):
                if p['name'] == g['name']:  # Only compare params for matching functions
                    similarity = self._compare_params(p['arguments'], g['arguments'])
                    if similarity > 0.5:  # Only count significant matches
                        param_scores.append(similarity)
            
            if param_scores:
                metrics['f1_param'] = sum(param_scores) / max(len(predicted), len(gold))
            
            # Partial sequence match - count correct function names in sequence
            min_len = min(len(predicted), len(gold))
            correct_calls = 0
            total_calls = max(len(predicted), len(gold))
            
            for i in range(min_len):
                p, g = predicted[i], gold[i]
                # For partial match, check both function name and parameters
                if p['name'] == g['name']:
                    param_similarity = self._compare_params(p['arguments'], g['arguments'])
                    if param_similarity > 0.5:  # Only count significant matches
                        correct_calls += 1
            
            metrics['partial_match'] = correct_calls / total_calls
            
            # Full sequence match - requires exact match of everything
            metrics['full_match'] = float(
                len(predicted) == len(gold) and 
                all(p['name'] == g['name'] and self._compare_params(p['arguments'], g['arguments']) > 0.9
                    for p, g in zip(predicted, gold))
            )
            
            # Win rate - handle both numeric and list results
            if executed_result is not None and gold_answer is not None:
                if isinstance(executed_result, (int, float)) and isinstance(gold_answer, (int, float)):
                    metrics['win_rate'] = float(abs(executed_result - gold_answer) < 1e-6)
                elif isinstance(executed_result, list) and isinstance(gold_answer, list):
                    metrics['win_rate'] = float(executed_result == gold_answer)
                else:
                    metrics['win_rate'] = float(executed_result == gold_answer)
            
            if self.debug:
                logger.info(f"Calculated metrics: {metrics}")
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            
        return metrics

    def _calculate_f1(self, predicted: List, gold: List) -> float:
        """Calculate F1 score between two lists"""
        if not predicted or not gold:
            return 0.0
        
        pred_set = set(predicted)
        gold_set = set(gold)
        
        true_positives = len(pred_set.intersection(gold_set))
        precision = true_positives / len(pred_set) if pred_set else 0
        recall = true_positives / len(gold_set) if gold_set else 0
        
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)

    def evaluate_sample(self, sample: NestfulRow) -> Dict[str, float]:
        """Evaluate a single sample"""
        try:
            # Prepare the prompt
            system_prompt = load_system_prompt(sample.tools)
            
            # Get model completion
            model_output = get_completion(
                self.model_name,
                self.provider,
                system_prompt,
                sample.input
            )
            
            # Parse the function sequence
            predicted_sequence = self.parse_function_sequence(model_output)
            if not predicted_sequence:
                return {
                    'f1_function': 0.0,
                    'f1_param': 0.0,
                    'partial_match': 0.0,
                    'full_match': 0.0,
                    'win_rate': 0.0
                }
            
            # Execute the sequence
            executed_result, _ = self.execute_function_sequence(predicted_sequence)
            
            # Calculate metrics
            return self.calculate_metrics(
                predicted_sequence,
                sample.output,
                executed_result,
                sample.gold_answer
            )
            
        except Exception as e:
            logger.error(f"Error evaluating sample: {str(e)}")
            return {
                'f1_function': 0.0,
                'f1_param': 0.0,
                'partial_match': 0.0,
                'full_match': 0.0,
                'win_rate': 0.0
            }

    def evaluate(self, batch_size: int = 8) -> Dict[str, float]:
        """Evaluate all samples in the dataset"""
        # Load evaluation data
        eval_data = load_data()
        
        # Initialize metrics
        total_metrics = {
            'f1_function': 0.0,
            'f1_param': 0.0,
            'partial_match': 0.0,
            'full_match': 0.0,
            'win_rate': 0.0
        }
        
        # Process samples
        for sample in tqdm(eval_data, desc="Evaluating samples"):
            metrics = self.evaluate_sample(sample)
            for k, v in metrics.items():
                total_metrics[k] += v
        
        # Calculate averages
        num_samples = len(eval_data)
        return {k: v/num_samples for k, v in total_metrics.items()}