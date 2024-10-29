import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

# Specify the model path
#model_name = "meta-llama/Llama-2-7b-chat-hf"
model_name = "meta-llama/Llama-3.1-8B-Instruct"
model_name = "meta-llama/Llama-3.1-70B-Instruct"


## Load the tokenizer and model with MPS (Apple Silicon) support
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("mps")


def chat():
    print("Chat with LLM. Type 'quit' to exit.")
    conversation_history = ""
    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        # Append user input to conversation history
        #conversation_history = f"User: {user_input}\n"
        conversation_history = f'<|begin_of_text|><|start_header_id|>{user_input}<|end_header_id|>\n\n{0}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nSure thing!'

        # Tokenize the input and move tensors to MPS
        inputs = tokenizer(conversation_history, return_tensors="pt").to("mps")
        streamer = TextStreamer(tokenizer)
        
        # Generate response word by word
        print("LLM: ", end="")
        model.generate(**inputs, max_new_tokens=512, streamer=streamer)

        ## Generate response
        #with torch.no_grad():
        #    outputs = model.generate(inputs.input_ids, max_length=512, pad_token_id=tokenizer.eos_token_id, max_new_tokens=256)

        # Decode and display the model's response
        #response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        #response_text = response[len(conversation_history):].strip()#

        #print(f"LLM: {response_text}")

        # Append model's response to conversation history
        #conversation_history += f"LLaMA: {response_text}\n"

# Start the chat

chat()



