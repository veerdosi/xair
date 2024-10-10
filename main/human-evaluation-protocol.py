class HumanEvaluationProtocol:
       @staticmethod
       def evaluate_explanation(question, answer, explanation, evaluator_id):
           print(f"Question: {question}")
           print(f"Model Answer: {answer}")
           print(f"Explanation: {explanation}")
           
           plausibility = int(input("Rate the plausibility of the explanation (1-5): "))
           faithfulness = int(input("Rate how well the explanation reflects the answer (1-5): "))
           coherence = int(input("Rate the coherence of the explanation (1-5): "))
           usefulness = int(input("Rate how useful the explanation is (1-5): "))
           
           return {
               "evaluator_id": evaluator_id,
               "plausibility": plausibility,
               "faithfulness": faithfulness,
               "coherence": coherence,
               "usefulness": usefulness
           }

   # Usage in main script:
   # human_eval = HumanEvaluationProtocol()
   # results = []
   # for sample in test_samples:
   #     eval_result = human_eval.evaluate_explanation(sample['question'], sample['answer'], sample['explanation'], evaluator_id)
   #     results.append(eval_result)
   