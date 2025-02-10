from evaluation.eval import NESTFULEvaluator
from evaluation.settings import SAVE_RESULTS

import logging
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Model configuration
    model_name = "google/gemini-2.0-flash-001"
    provider = "openrouter"
    
    # Initialize evaluator
    evaluator = NESTFULEvaluator(
        model_name=model_name,
        provider=provider,
        debug=True
    )
    
    try:
        # Run evaluation
        logger.info(f"Starting evaluation of {model_name} with provider {provider}")
        results = evaluator.evaluate(batch_size=1)  # Using small batch size for toy data
        
        # Print results
        print("\nEvaluation Results:")
        for metric_name, value in results.items():
            print(f"{metric_name}: {value:.4f}")
            
        # Save results to file
        if SAVE_RESULTS:
            output_file = "evaluation_results.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_file}")
            
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
