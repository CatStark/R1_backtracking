import pandas as pd
import datasets
from vllm import LLM, SamplingParams
from tqdm.notebook import tqdm
from datetime import datetime
from google.colab import drive


class GSM8KDataCollector:
    def __init__(self):
        # Mount Google Drive
        try:
            drive.mount('/content/drive')
        except Exception as e:
            print(f"Warning: Could not mount Google Drive. Error: {e}")

        # Initialize model
        self.model = LLM(
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            trust_remote_code=True,
            dtype="float16",
        )

        # Define sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=3512,
            top_p=0.9,
        )

    def get_math_problems(self, n_samples: int = 100) -> tuple:
        """Load GSM8K dataset samples with formatted prompts"""
        dataset = datasets.load_dataset("gsm8k", "main", split="test")
        problems = dataset[:n_samples]["question"]
        answers = dataset[:n_samples]["answer"]

        prompt_template = """Solve this problem.

Problem:
{problem}
After you've solved it, indicate your final numerical answer using this exact format:
### ANSWER: X
where X is your final numerical answer without units."""

        formatted_problems = [
            prompt_template.format(problem=problem)
            for problem in problems
        ]

        return formatted_problems, answers

    def collect_data(self, n_problems: int = 20):
        """Collect model responses for math problems"""
        # Get problems and answers
        problems, answers = self.get_math_problems(n_problems)

        # Store results
        results = []

        # Process each problem
        for idx, (problem, answer) in enumerate(tqdm(zip(problems, answers), total=len(problems))):
            try:
                # Generate response
                output = self.model.generate([problem], self.sampling_params)[0].outputs[0].text

                # Store raw data
                results.append({
                    "gsm8k_id": f"Q{idx + 1}",
                    "question": problem,
                    "full_response": output,
                    "answer": answer
                })
            except Exception as e:
                print(f"Error processing problem {idx + 1}: {e}")
                continue

        # Convert to DataFrame and save
        df = pd.DataFrame(results)
        self.save_results(df)

        return df

    def save_results(self, df: pd.DataFrame):
        """Save raw results to Google Drive"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_path = "/content/drive/MyDrive/GSM8K_Data"

        try:
            import os
            os.makedirs(folder_path, exist_ok=True)

            # Save raw results
            df.to_csv(f"{folder_path}/raw_responses_{timestamp}.csv", index=False)
            print(f"\nResults saved to: raw_responses_{timestamp}.csv")

        except Exception as e:
            print(f"Error saving results: {e}")


# Usage
if __name__ == "__main__":
    collector = GSM8KDataCollector()
    results = collector.collect_data(n_problems=200)  # Adjust as needed
    print("\nData collection completed!")