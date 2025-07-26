# agent_sdk.py

import os
import sys
import json
import argparse
import logging
from textwrap import indent

# --- SDK IMPORTS ---
from google import genai
from google.genai import types
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

api_key = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-2.5-pro"


# --- LOGGING CONFIGURATION ---
_log_file = None
original_print = print

def log_print(*args, **kwargs):
    """
    自定义打印函数，同时输出到 stdout 和日志文件。
    """
    original_print(*args, **kwargs)
    if _log_file is not None:
        message = ' '.join(str(arg) for arg in args)
        _log_file.write(message + '\n')
        _log_file.flush()

print = log_print

def set_log_file(log_file_path):
    """设置日志文件路径。"""
    global _log_file
    if log_file_path:
        try:
            _log_file = open(log_file_path, 'w', encoding='utf-8')
            return True
        except Exception as e:
            print(f"Error opening log file {log_file_path}: {e}")
            return False
    return True

def close_log_file():
    """关闭日志文件。"""
    global _log_file
    if _log_file is not None:
        _log_file.close()
        _log_file = None

# --- PROMPTS (Unchanged from original) ---
step1_prompt = """
### Core Instructions ###

*   **Rigor is Paramount:** Your primary goal is to produce a complete and rigorously justified solution. Every step in your solution must be logically sound and clearly explained. A correct final answer derived from flawed or incomplete reasoning is considered a failure.
*   **Honesty About Completeness:** If you cannot find a complete solution, you must **not** guess or create a solution that appears correct but contains hidden flaws or justification gaps. Instead, you should present only significant partial results that you can rigorously prove. A partial result is considered significant if it represents a substantial advancement toward a full solution. Examples include:
    *   Proving a key lemma.
    *   Fully resolving one or more cases within a logically sound case-based proof.
    *   Establishing a critical property of the mathematical objects in the problem.
    *   For an optimization problem, proving an upper or lower bound without proving that this bound is achievable.
*   **Use TeX for All Mathematics:** All mathematical variables, expressions, and relations must be enclosed in TeX delimiters (e.g., `Let $n$ be an integer.`).

### Output Format ###

Your response MUST be structured into the following sections, in this exact order.

**1. Summary**

Provide a concise overview of your findings. This section must contain two parts:

*   **a. Verdict:** State clearly whether you have found a complete solution or a partial solution.
    *   **For a complete solution:** State the final answer, e.g., "I have successfully solved the problem. The final answer is..."
    *   **For a partial solution:** State the main rigorous conclusion(s) you were able to prove, e.g., "I have not found a complete solution, but I have rigorously proven that..."
*   **b. Method Sketch:** Present a high-level, conceptual outline of your solution. This sketch should allow an expert to understand the logical flow of your argument without reading the full detail. It should include:
    *   A narrative of your overall strategy.
    *   The full and precise mathematical statements of any key lemmas or major intermediate results.
    *   If applicable, describe any key constructions or case splits that form the backbone of your argument.

**2. Detailed Solution**

Present the full, step-by-step mathematical proof. Each step must be logically justified and clearly explained. The level of detail should be sufficient for an expert to verify the correctness of your reasoning without needing to fill in any gaps. This section must contain ONLY the complete, rigorous proof, free of any internal commentary, alternative approaches, or failed attempts.

### Self-Correction Instruction ###

Before finalizing your output, carefully review your "Method Sketch" and "Detailed Solution" to ensure they are clean, rigorous, and strictly adhere to all instructions provided above. Verify that every statement contributes directly to the final, coherent mathematical argument.

"""

self_improvement_prompt = """
You have an opportunity to improve your solution. Please review your solution carefully. Correct errors and fill justification gaps if any. Your second round of output should strictly follow the instructions in the system prompt.
"""

correction_prompt = """
Below is the bug report. If you agree with certain item in it, can you improve your solution so that it is complete and rigorous? Note that the evaluator who generates the bug report can misunderstand your solution and thus make mistakes. If you do not agree with certain item in the bug report, please add some detailed explanations to avoid such misunderstanding. Your new solution should strictly follow the instructions in the system prompt.
"""

verification_system_prompt = """
You are an expert mathematician and a meticulous grader for an International Mathematical Olympiad (IMO) level exam. Your primary task is to rigorously verify the provided mathematical solution. A solution is to be judged correct **only if every step is rigorously justified.** A solution that arrives at a correct final answer through flawed reasoning, educated guesses, or with gaps in its arguments must be flagged as incorrect or incomplete.

### Instructions ###

**1. Core Instructions**
*   Your sole task is to find and report all issues in the provided solution. You must act as a **verifier**, NOT a solver. **Do NOT attempt to correct the errors or fill the gaps you find.**
*   You must perform a **step-by-step** check of the entire solution. This analysis will be presented in a **Detailed Verification Log**, where you justify your assessment of each step: for correct steps, a brief justification suffices; for steps with errors or gaps, you must provide a detailed explanation.

**2. How to Handle Issues in the Solution**
When you identify an issue in a step, you MUST first classify it into one of the following two categories and then follow the specified procedure.

*   **a. Critical Error:**
    This is any error that breaks the logical chain of the proof. This includes both **logical fallacies** (e.g., claiming that `A>B, C>D` implies `A-C>B-D`) and **factual errors** (e.g., a calculation error like `2+3=6`).
    *   **Procedure:**
        *   Explain the specific error and state that it **invalidates the current line of reasoning**.
        *   Do NOT check any further steps that rely on this error.
        *   You MUST, however, scan the rest of the solution to identify and verify any fully independent parts. For example, if a proof is split into multiple cases, an error in one case does not prevent you from checking the other cases.

*   **b. Justification Gap:**
    This is for steps where the conclusion may be correct, but the provided argument is incomplete, hand-wavy, or lacks sufficient rigor.
    *   **Procedure:**
        *   Explain the gap in the justification.
        *   State that you will **assume the step's conclusion is true** for the sake of argument.
        *   Then, proceed to verify all subsequent steps to check if the remainder of the argument is sound.

**3. Output Format**
Your response MUST be structured into two main sections: a **Summary** followed by the **Detailed Verification Log**.

*   **a. Summary**
    This section MUST be at the very beginning of your response. It must contain two components:
    *   **Final Verdict**: A single, clear sentence declaring the overall validity of the solution. For example: "The solution is correct," "The solution contains a Critical Error and is therefore invalid," or "The solution's approach is viable but contains several Justification Gaps."
    *   **List of Findings**: A bulleted list that summarizes **every** issue you discovered. For each finding, you must provide:
        *   **Location:** A direct quote of the key phrase or equation where the issue occurs.
        *   **Issue:** A brief description of the problem and its classification (**Critical Error** or **Justification Gap**).

*   **b. Detailed Verification Log**
    Following the summary, provide the full, step-by-step verification log as defined in the Core Instructions. When you refer to a specific part of the solution, **quote the relevant text** to make your reference clear before providing your detailed analysis of that part.

**Example of the Required Summary Format**
*This is a generic example to illustrate the required format. Your findings must be based on the actual solution provided below.*

**Final Verdict:** The solution is **invalid** because it contains a Critical Error.

**List of Findings:**
*   **Location:** "By interchanging the limit and the integral, we get..."
    *   **Issue:** Justification Gap - The solution interchanges a limit and an integral without providing justification, such as proving uniform convergence.
*   **Location:** "From $A > B$ and $C > D$, it follows that $A-C > B-D$"
    *   **Issue:** Critical Error - This step is a logical fallacy. Subtracting inequalities in this manner is not a valid mathematical operation.

"""

verification_remider = """
### Verification Task Reminder ###

Your task is to act as an IMO grader. Now, generate the **summary** and the **step-by-step verification log** for the solution above. In your log, justify each correct step and explain in detail any errors or justification gaps you find, as specified in the instructions above.
"""

# --- REFACTORED API CALL FUNCTION ---

def call_gemini_api(
    system_instruction: str | None,
    contents: list,
    verbose: bool = True
) -> str | None:
    """
    使用 SDK 调用 Gemini API，启用流式传输和思考过程。
    
    Args:
        system_instruction: 系统提示词。
        contents: 用户/模型的对话回合列表。
        verbose: 是否打印思考过程和 token 使用情况。

    Returns:
        生成的文本内容，如果失败则返回 None。
    """
    if verbose:
        print("--- [Calling Gemini API via SDK] ---")

    GENERATION_CONFIG = types.GenerateContentConfig(
        temperature=0.1,
        top_p=1.0, 
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,
            thinking_budget=32768
        ),
        system_instruction=system_instruction
    )

    try:
        if verbose:
            print("🧠 Thinking process:")
            print("💭 ", end='', flush=True)

        full_answer = ""
        thinking_shown = False
        final_usage_metadata = None

        response_stream = client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=contents,
            config=GENERATION_CONFIG
        )

        for chunk in response_stream:
            if chunk.usage_metadata:
                final_usage_metadata = chunk.usage_metadata

            if not chunk.candidates:
                continue
            
            content_obj = chunk.candidates[0].content
            if not content_obj or not getattr(content_obj, "parts", None):
                continue

            for part in content_obj.parts:
                if part.thought:
                    if not thinking_shown:
                        print()
                        thinking_shown = True
                    print(f"🤔 {part.text}", end='', flush=True)
                elif part.text:
                    if thinking_shown:
                        print("\n\n📝 Answer:")
                        thinking_shown = False
                    print(part.text, end='', flush=True)
                    full_answer += part.text
        
        print() # 确保在流式输出后换行

        if verbose and final_usage_metadata:
            print(f"\nThinking Tokens: {final_usage_metadata.thoughts_token_count}")
            print(f"Answer Tokens: {final_usage_metadata.candidates_token_count}")
            print(f"Prompt Tokens: {final_usage_metadata.prompt_token_count}")
            print(f"Total Tokens: {final_usage_metadata.total_token_count}")
        
        if verbose:
            print("--- [API Call Completed] ---\n")
        
        return full_answer if full_answer else None

    except Exception as e:
        print(f"\nSDK API call failed: \n{e}")
        return None

# --- HELPER & LOGIC FUNCTIONS (Refactored) ---

def read_file_content(filepath):
    """
    读取并返回文件内容。
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{filepath}': {e}")
        sys.exit(1)

def extract_detailed_solution(solution, marker='Detailed Solution', after=True):
    """
    从解决方案字符串中提取 '### Detailed Solution ###' 之后或之前的文本。
    """
    idx = solution.find(marker)
    if idx == -1:
        return ''
    if after:
        return solution[idx + len(marker):].strip()
    else:
        return solution[:idx].strip()

def verify_solution(problem_statement, solution, verbose=True):
    dsol = extract_detailed_solution(solution)
    newst = f"""
======================================================================
### Problem ###

{problem_statement}

======================================================================
### Solution ###

{dsol}

{verification_remider}
"""
    if verbose:
        print(">>>>>>> Start verification.")
    
    contents1 = [{"role": "user", "parts": [{"text": newst}]}]
    out = call_gemini_api(
        system_instruction=verification_system_prompt, 
        contents=contents1,
        verbose=verbose
    )
    if not out:
        print(">>>>>>> Verification call failed.")
        return "", "no"

    if verbose:
        print(">>>>>>> Verification results:")
        print(out)

    check_correctness = f'Response in "yes" or "no". Is the following statement saying the solution is correct, or does not contain critical error or a major justification gap?\n\n{out}'
    contents2 = [{"role": "user", "parts": [{"text": check_correctness}]}]
    o = call_gemini_api(
        system_instruction=None,
        contents=contents2,
        verbose=verbose
    )
    if not o:
        print(">>>>>>> Verification check call failed.")
        return "", "no"

    if verbose:
        print(">>>>>>> Is verification good?")
        print(o)
        
    bug_report = ""
    if "yes" not in o.lower():
        bug_report = extract_detailed_solution(out, "Detailed Verification", False)

    if verbose:
        print(">>>>>>> Bug report:")
        print(bug_report)
    
    return bug_report, o

def check_if_solution_claimed_complete(solution):
    check_complete_prompt = f"""
Is the following text claiming that the solution is complete?
==========================================================

{solution}

==========================================================

Response in exactly "yes" or "no". No other words.
    """
    contents = [{"role": "user", "parts": [{"text": check_complete_prompt}]}]
    o = call_gemini_api(
        system_instruction=None,
        contents=contents,
        verbose=False # This is a simple check, no need for verbose output
    )
    if not o:
        return False
        
    print(o)
    return "yes" in o.lower()

def init_explorations(problem_statement, verbose=True, other_prompts=[]):
    contents = [{"role": "user", "parts": [{"text": problem_statement}]}]
    if other_prompts:
        for prompt in other_prompts:
            contents.append({"role": "user", "parts": [{"text": prompt}]})

    print(">>>>>> Initial prompt.")
    output1 = call_gemini_api(
        system_instruction=step1_prompt,
        contents=contents,
        verbose=verbose
    )
    if not output1:
        print(">>>>>> Initial generation failed.")
        return None, None, None, None
        
    print(">>>>>>> First solution: ") 
    print(output1)

    print(">>>>>>> Self improvement start:")
    contents.append({"role": "model", "parts": [{"text": output1}]})
    contents.append({"role": "user", "parts": [{"text": self_improvement_prompt}]})
    
    solution = call_gemini_api(
        system_instruction=step1_prompt,
        contents=contents,
        verbose=verbose
    )
    if not solution:
        print(">>>>>> Self-improvement generation failed.")
        return None, None, None, None

    print(">>>>>>> Corrected solution: ")
    print(solution)
    
    print(">>>>>>> Check if solution is complete:")
    is_complete = check_if_solution_claimed_complete(solution)
    if not is_complete:
        print(">>>>>>> Solution is not complete. Failed.")
        return None, None, None, None
    
    print(">>>>>>> Verify the solution.")
    verify, good_verify = verify_solution(problem_statement, solution, verbose)

    print(">>>>>>> Initial verification: ")
    print(verify)
    print(f">>>>>>> verify results: {good_verify}")
    
    return True, solution, verify, good_verify # Return a success flag instead of p1

def agent(problem_statement, other_prompts=[]):
    success_flag, solution, verify, good_verify = init_explorations(problem_statement, True, other_prompts)

    if not success_flag or solution is None:
        print(">>>>>>> Failed in finding a complete solution during initialization.")
        return None

    error_count = 0
    correct_count = 1 if "yes" in good_verify.lower() else 0
    
    for i in range(30):
        print(f"Number of iterations: {i}, number of corrects: {correct_count}, number of errors: {error_count}")

        if "yes" not in good_verify.lower():
            correct_count = 0
            error_count += 1

            print(">>>>>>> Verification does not pass, correcting ...")
            
            # 建立一个新的对话历史用于修正
            correction_contents = [{"role": "user", "parts": [{"text": problem_statement}]}]
            if other_prompts:
                 for prompt in other_prompts:
                    correction_contents.append({"role": "user", "parts": [{"text": prompt}]})
            
            correction_contents.append({"role": "model", "parts": [{"text": solution}]})
            correction_contents.append({"role": "user", "parts": [{"text": correction_prompt}, {"text": verify}]})

            print(">>>>>>> New correction prompt being sent.")
            new_solution = call_gemini_api(
                system_instruction=step1_prompt,
                contents=correction_contents,
                verbose=True
            )
            
            if not new_solution:
                print(">>>>>>> Correction attempt failed to generate a response. Stopping.")
                return None
            
            solution = new_solution
            print(">>>>>>> Corrected solution:")
            print(solution)

            print(">>>>>>> Check if new solution is complete:")
            is_complete = check_if_solution_claimed_complete(solution)
            if not is_complete:
                print(">>>>>>> Solution is not complete after correction. Failed.")
                return None
        
        print(">>>>>>> Verify the solution.")
        verify, good_verify = verify_solution(problem_statement, solution)

        if "yes" in good_verify.lower():
            print(">>>>>>> Solution is good, verifying again ...")
            correct_count += 1
            error_count = 0
        else:
             # 如果验证再次失败，重置正确计数器
             correct_count = 0

        if correct_count >= 5:
            print(">>>>>>> Correct solution found and verified multiple times.")
            print(solution)
            return solution
        elif error_count >= 10:
            print(">>>>>>> Failed to find a correct solution after multiple errors.")
            return None

    print(">>>>>>> Failed to find a correct solution within the iteration limit.")
    return None

# --- MAIN EXECUTION BLOCK (Unchanged from original) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IMO Problem Solver Agent (SDK Version)')
    parser.add_argument('problem_file', nargs='?', default='problem_statement.txt', 
                       help='Path to the problem statement file (default: problem_statement.txt)')
    parser.add_argument('--log', '-l', type=str, help='Path to log file (optional)')
    parser.add_argument('--other_prompts', '-o', type=str, help='Comma-separated other prompts (optional)')
    parser.add_argument("--max_runs", '-m', type=int, default=10, help='Maximum number of runs (default: 10)')
    
    args = parser.parse_args()

    max_runs = args.max_runs
    
    other_prompts = []
    if args.other_prompts:
        other_prompts = args.other_prompts.split(',')

    print(">>>>>>> Other prompts:")
    print(other_prompts)

    if args.log:
        if not set_log_file(args.log):
            sys.exit(1)
        print(f"Logging to file: {args.log}")
    
    problem_statement = read_file_content(args.problem_file)

    for i in range(max_runs):
        print(f"\n\n>>>>>>>>>>>>>>>>>>>>>>>>>> Run {i+1} of {max_runs} ...")
        try:
            sol = agent(problem_statement, other_prompts)
            if sol is not None:
                print(f">>>>>>> Found a correct solution in run {i+1}.")
                # 最终解决方案会以清晰的格式打印在 agent 函数的末尾
                # print(sol) # 可以取消注释以再次打印
                break
        except Exception as e:
            print(f">>>>>>> Error in run {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    close_log_file()