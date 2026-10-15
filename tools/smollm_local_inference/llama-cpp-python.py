# [MODULE DOC] Minimal example: downloads the SmolLM2-1.7B-Instruct q4_k_m GGUF from the Hub with
# llama-cpp-python, runs a
# raw completion for a planets question (max 32 tokens, stops at "Q:" or newline, echoes the prompt) and
# prints the full output mapping. Defines no functions.
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF",
    filename="*q4_k_m.gguf",
    verbose=False
)

output = llm(
      "Q: Name the planets in the solar system? A: ", # Prompt
      max_tokens=32, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion

print(output)