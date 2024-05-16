import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Define your model path
model_path = "./merged"  # or the path/model_name you have

# Your custom quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    quantization_config=quantization_config,
    output_hidden_states=False,  # Disable output_hidden_states
)
tokenizer = AutoTokenizer.from_pretrained(model_path)


def generate_text(input_text: str) -> str:
    """
    generate_text - generate text using the model

    :param str input_text: input text
    :return str: generated text
    """
    # Encode the input text
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Generate a sequence of tokens in response to the input text
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        top_k=4,
        penalty_alpha=0.6,
        num_return_sequences=1,
    )

    # Decode the generated tokens to a readable text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text


# Example usage
input_text = "The future of AI is"
generated_text = generate_text(input_text)
print(generated_text)
