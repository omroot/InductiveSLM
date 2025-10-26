#!/usr/bin/env python3
"""
Example script demonstrating how to use the OpenAI-based semantic evaluation module.

This script shows how to:
1. Use eval_semantic_similarity_openai to compare predictions with ground truth
2. Save the evaluation results
3. Compute accuracy based on semantic similarity
"""

from src.models.evaluate import (
    eval_semantic_similarity_openai,
    save_semantic_evaluation_results
)


def main():
    # Example predictions and ground truth
    predictions = [
        "The capital of France is Paris.",
        "Einstein developed the theory of relativity.",
        "Water boils at 100 degrees Celsius.",
        "The Earth orbits around the Sun.",
        "Python is a programming language."
    ]

    ground_truth = [
        "Paris is the capital city of France.",
        "The theory of relativity was created by Albert Einstein.",
        "At sea level, water boils at 100°C.",
        "Our planet Earth revolves around the Sun.",
        "Java is an object-oriented programming language."  # This one is different!
    ]

    print("=" * 80)
    print("OpenAI Semantic Similarity Evaluation Example")
    print("=" * 80)

    # Run semantic evaluation using OpenAI
    results = eval_semantic_similarity_openai(
        preds=predictions,
        refs=ground_truth,
        model="gpt-4o-mini",  # You can also use "gpt-4o", "gpt-3.5-turbo", etc.
        temperature=0.0
    )

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Total examples: {results['total']}")
    print(f"Correct matches: {results['correct']}")
    print(f"Accuracy: {results['accuracy']:.2%}")

    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)
    for i, result in enumerate(results['results'], 1):
        print(f"\nExample {i}:")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Reference:  {result['reference']}")
        print(f"  Evaluation: {result['evaluation']}")
        print(f"  Correct:    {result['is_correct']}")

    # Save results to files
    output_dir = "./evaluation_output"
    save_semantic_evaluation_results(
        output_dir=output_dir,
        filename="semantic_eval_example",
        evaluation_results=results
    )

    print("\n" + "=" * 80)
    print(f"Results saved to {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
