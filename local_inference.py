# Example of local inference for language model
# First download GGUF model weights from the Bloke ([model].Q4_K_M.gguf)
# Tested for Zephyr, MistralOpenOrca, and Intel NeuralChat

from llama_cpp import Llama
import os

# Code to display results interactively. Ignore outside of a notebook:
from IPython.display import Markdown, display
def display_response(response) -> None:
    """Display response for jupyter notebook."""
    if response is None:
        response_text = "None"
    else:
    	# This may vary if using a different model
        response_text = response['choices'][0]['text'].strip()

    display(Markdown(f"**`Final Response:`** {response_text}"))
    
 # Prompt formatting. Add the format for the model you choose.
 def format_prompt(prompt: str, model_name: str) -> str:
    """Format prompt for each of three model. 
    Input: text string and model name. 
    Output: formatted text string ready to serve as LLM input, in the following format:
    Mistral orca prompt format: "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant"
    Zephyr prompt format: "<|user|>\n{prompt}</s>\n<|assistant|>"
    Intel prompt format: "### User:\n{prompt}\n### Assistant:\n"
    """
    if model_name == "mistralorca":
        formatted_prompt = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant"
    elif model_name == "zephyr":
        formatted_prompt = "<|user|>\n" + prompt + "</s>\n<|assistant|>"
    elif model_name == "intel":
        formatted_prompt = "### User:\n" + prompt + "\n### Assistant:\n"
    else:
        print("Invalid model name. Options are 'mistralorca', 'zephyr', or 'intel'.")
    return formatted_prompt
 
 # Load the model
 myllama = Llama(
    model_path='/path/to/modelweights/modelname.Q4_K_M.gguf',
    n_gpu_layers=-1
)

# Inference with mistralorca prompt format
output = myllama(
  format_prompt(pre_prompt + " " + prompt + " " + post_prompt, "mistralorca"), 
  max_tokens=300,
  #stop=["<|im_start|>", "\n"], # Stop on this token
  echo=False # Echo the prompt back in the output
)
# Display answer if in notebook:
display_response(output)