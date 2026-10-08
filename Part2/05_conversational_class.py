from llama_cpp import Llama

# Path to the downloaded GGUF model file.
llama_path = "./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"

# Initialize the model
llm = Llama(model_path=llama_path, verbose=False)


class Conversation:
    # Complete the __init__ method of the Conversation class
    def __init__(self, llm: Llama, system_prompt='', history=[]):
        self.llm = llm
        self.system_prompt = system_prompt
        self.history = [{"role": "system", "content": self.system_prompt}] + history

    def create_completion(self, user_prompt=''):
        # Add the user prompt to the history
        self.history.append({"role": "user", "content": user_prompt})
        # Send the history messages to the LLM
        output = self.llm.create_chat_completion(messages=self.history)
        conversation_result = output['choices'][0]['message']
        # Append the conversation_result to the history
        self.history.append(conversation_result)
        return conversation_result['content']

## Single Conversation
instruction = "You are a travel expert that recommends a travel destination based on a specification. Return the location name only in City, Country form."

# Define a chatbot using the Conversation class
chatbot = Conversation(llm, system_prompt=instruction)

# Send a prompt to the model
result = chatbot.create_completion("I'd like to learn about the Aztecs.")
print(result)

# Multiple Conversation
chatbot = Conversation(llm, system_prompt="You are a travel expert that recommends a travel destination based on a prompt. Return the location name only as 'City, Country'.")

# Ask for the initial travel recommendation
first_recommendation = chatbot.create_completion("Recommend a Spanish-speaking city.")
print(first_recommendation)

# Add an additional request to update the recommendation
second_recommendation = chatbot.create_completion("A different city in the same country")
print(second_recommendation)
