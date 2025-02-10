from typing import List, Dict, Any, Optional, Tuple
import json
import logging
from tqdm import tqdm
import argparse
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

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
        self.executor = ThreadPoolExecutor(max_workers=8)  # For CPU-bound tasks
        logger.info(f"Initialized evaluator with model: {model_name}")

    def parse_function_sequence(self, model_output: str) -> List[Dict]:
        """Parse function sequence from model output"""
        try:
            # First try to parse the entire output as JSON
            try:
                sequence = json.loads(model_output)
                if isinstance(sequence, list):
                    logger.info(f"Parsed function sequence directly: {str(sequence)[:100]}...")
                    return sequence
            except json.JSONDecodeError:
                pass

            # Look for code blocks with JSON content
            code_blocks = re.findall(r'```(?:json)?\s*(.*?)```', model_output, re.DOTALL)
            if not code_blocks:
                # Try to find JSON array without code blocks
                matches = re.findall(r'\[(.*?)\]', model_output, re.DOTALL)
                if matches:
                    code_blocks = [f"[{match}]" for match in matches]

            for block in code_blocks:
                try:
                    sequence = json.loads(block)
                    if isinstance(sequence, list):
                        logger.info(f"Parsed function sequence from json code block: {str(sequence)[:100]}...")
                        return sequence
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.warning(f"Error parsing json block: {str(e)}")
                    continue
            
            logger.warning("No valid JSON found in json code blocks")
            return []
            
        except Exception as e:
            logger.error(f"Error parsing function sequence: {str(e)}")
            return []

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
            # For variable references, normalize to standard format
            if value.startswith('$') and value.endswith('$'):
                parts = value[1:-1].split('.')
                return f"${parts[0]}.{parts[1]}$" if len(parts) > 1 else f"${parts[0]}$"
            return value.lower()
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
        
        total_matches = 0
        total_params = len(gold_args)
        
        for key in gold_args:
            if key not in pred_args:
                continue
                
            pred_val = self._normalize_param(pred_args[key])
            gold_val = self._normalize_param(gold_args[key])
            
            # Exact match for variable references
            if ('$' in pred_val and '$' in gold_val):
                if pred_val == gold_val:
                    total_matches += 1
            # Numeric comparison with tolerance
            elif isinstance(pred_args[key], (int, float)) and isinstance(gold_args[key], (int, float)):
                if abs(float(pred_args[key]) - float(gold_args[key])) < 1e-6:
                    total_matches += 1
            # String comparison
            elif pred_val == gold_val:
                total_matches += 1
                
        return total_matches / total_params if total_params > 0 else 0.0

    def _calculate_f1(self, predicted: List[Dict], gold: List[Dict], compare_params: bool = False) -> float:
        """Calculate F1 score between predicted and gold sequences"""
        if not predicted or not gold:
            return 0.0
            
        true_positives = 0
        for p in predicted:
            for g in gold:
                if p['name'] == g['name']:
                    if not compare_params:
                        true_positives += 1
                        break
                    else:
                        param_similarity = self._compare_params(p['arguments'], g['arguments'])
                        if param_similarity > 0.8:  # High threshold for parameter match
                            true_positives += param_similarity
                            break
                            
        precision = true_positives / len(predicted) if predicted else 0
        recall = true_positives / len(gold) if gold else 0
        
        return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

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
                
            # Convert gold to list of dicts if needed
            if isinstance(gold[0], ToolCall):
                gold = [{'name': g.name, 'label': g.label, 'arguments': g.arguments} for g in gold]
            
            # F1 for function names only
            metrics['f1_function'] = self._calculate_f1(predicted, gold, compare_params=False)
            
            # F1 for parameters
            metrics['f1_param'] = self._calculate_f1(predicted, gold, compare_params=True)
            
            # Partial sequence match
            correct_sequence = 0
            for i, (p, g) in enumerate(zip(predicted[:len(gold)], gold)):
                if p['name'] == g['name']:
                    param_similarity = self._compare_params(p['arguments'], g['arguments'])
                    if param_similarity > 0.8:
                        correct_sequence += 1
            
            metrics['partial_match'] = correct_sequence / len(gold) if gold else 0
            
            # Full sequence match - requires exact match of everything
            sequence_match = (len(predicted) == len(gold))
            if sequence_match:
                for p, g in zip(predicted, gold):
                    if p['name'] != g['name'] or self._compare_params(p['arguments'], g['arguments']) < 0.9:
                        sequence_match = False
                        break
            metrics['full_match'] = float(sequence_match)
            
            # Win rate
            if executed_result is not None and gold_answer is not None:
                if isinstance(executed_result, (int, float)) and isinstance(gold_answer, (int, float)):
                    metrics['win_rate'] = float(abs(executed_result - gold_answer) < 1e-6)
                else:
                    metrics['win_rate'] = float(executed_result == gold_answer)
            
            if self.debug:
                logger.info(f"Calculated metrics: {metrics}")
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            
        return metrics

    async def evaluate_batch(self, batch: List[NestfulRow]) -> List[Dict[str, float]]:
        """Evaluate a batch of samples concurrently"""
        # Prepare system prompts for the batch
        system_prompts = [load_system_prompt(sample.tools) for sample in batch]
        
        # Get model completions for the entire batch
        model_outputs = await get_completions_batch({
            "model": self.model_name,
            "provider": self.provider,
            "system_prompts": system_prompts,
            "user_prompts": [sample.input for sample in batch]
        })
        
        # Process each sample in the batch concurrently using ThreadPoolExecutor
        tasks = []
        for sample, model_output in zip(batch, model_outputs):
            # Parse function sequence
            predicted_sequence = self.parse_function_sequence(model_output)
            if not predicted_sequence:
                tasks.append({
                    'f1_function': 0.0,
                    'f1_param': 0.0,
                    'partial_match': 0.0,
                    'full_match': 0.0,
                    'win_rate': 0.0
                })
                continue
            
            # Execute sequence and calculate metrics
            executed_result, _ = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.execute_function_sequence,
                predicted_sequence
            )
            
            metrics = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                self.calculate_metrics,
                predicted_sequence,
                sample.output,
                executed_result,
                sample.gold_answer
            )
            tasks.append(metrics)
        
        return tasks

    async def evaluate(self, batch_size: int = 8) -> Dict[str, float]:
        """Evaluate all samples in the dataset using batched processing"""
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
        
        # Process samples in batches
        batches = [list(islice(eval_data, i, i + batch_size)) 
                  for i in range(0, len(eval_data), batch_size)]
        
        with tqdm(total=len(eval_data), desc="Evaluating samples") as pbar:
            for batch in batches:
                batch_metrics = await self.evaluate_batch(batch)
                
                # Aggregate batch results
                for metrics in batch_metrics:
                    for k, v in metrics.items():
                        total_metrics[k] += v
                
                pbar.update(len(batch))
        
        # Calculate averages
        num_samples = len(eval_data)
        return {k: v/num_samples for k, v in total_metrics.items()}