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

    def parse_function_sequence(self, model_output: str) -> List[Dict]:
        """Parse the model's output into a structured function sequence"""
        try:
            if self.debug:
                logger.info(f"Parsing model output: {model_output}")
                
            # Find the JSON list in the output
            start_idx = model_output.find('[')
            end_idx = model_output.rfind(']') + 1
            if start_idx == -1 or end_idx == 0:
                logger.warning("No JSON list found in model output")
                return []
            
            json_str = model_output[start_idx:end_idx]
            sequence = json.loads(json_str)
            
            # Validate sequence format
            for call in sequence:
                if not all(k in call for k in ['name', 'label', 'arguments']):
                    logger.warning(f"Invalid function call format: {call}")
                    return []
            
            if self.debug:
                logger.info(f"Parsed function sequence: {sequence}")
                
            return sequence
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
                args = {}
                for arg_name, arg_value in func_call['arguments'].items():
                    if isinstance(arg_value, str) and arg_value.startswith('$') and arg_value.endswith('$'):
                        # Handle variable reference
                        var_ref = arg_value[1:-1]
                        var_name, param = var_ref.split('.')
                        if var_name not in variables:
                            raise ValueError(f"Unknown variable reference: {var_name}")
                        if self.debug:
                            logger.info(f"Resolving variable reference {var_ref} to {variables[var_name][param]}")
                        args[arg_name] = variables[var_name][param]
                    else:
                        args[arg_name] = arg_value
                
                if self.debug:
                    logger.info(f"Function arguments: {args}")
                
                # Execute function
                result = func(**args)
                
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

    def calculate_metrics(
        self,
        predicted: List[Dict],
        gold: List[Dict],
        executed_result: Optional[float],
        gold_answer: float
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
            # F1 for function names
            pred_funcs = [call['name'] for call in predicted]
            gold_funcs = [call['name'] for call in gold]
            metrics['f1_function'] = self._calculate_f1(pred_funcs, gold_funcs)
            
            # F1 for parameters
            pred_params = []
            gold_params = []
            for p, g in zip(predicted, gold):
                pred_params.extend([f"{p['name']}_{k}_{v}" for k, v in p['arguments'].items()])
                gold_params.extend([f"{g['name']}_{k}_{v}" for k, v in g['arguments'].items()])
            metrics['f1_param'] = self._calculate_f1(pred_params, gold_params)
            
            # Partial sequence match
            correct_calls = sum(1 for p, g in zip(predicted, gold) 
                              if p['name'] == g['name'] and p['arguments'] == g['arguments'])
            metrics['partial_match'] = correct_calls / max(len(predicted), len(gold))
            
            # Full sequence match
            metrics['full_match'] = float(predicted == gold)
            
            # Win rate
            if executed_result is not None:
                metrics['win_rate'] = float(abs(executed_result - gold_answer) < 1e-6)
            
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

def main():
    """Main function to run the evaluation"""
    parser = argparse.ArgumentParser(description='Run NESTFUL evaluation')
    parser.add_argument('--model', required=True, help='Name of the model to evaluate')
    parser.add_argument('--provider', required=True, help='Provider to use (lm_studio, ollama, vllm, openrouter)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = NESTFULEvaluator(
        model_name=args.model,
        provider=args.provider,
        debug=args.debug
    )
    
    try:
        # Run evaluation
        logger.info(f"Starting evaluation of {args.model} with provider {args.provider}")
        results = evaluator.evaluate(batch_size=args.batch_size)
        
        # Print results
        print("\nEvaluation Results:")
        for metric_name, value in results.items():
            print(f"{metric_name}: {value:.4f}")
            
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
