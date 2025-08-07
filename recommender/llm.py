import openai
import os

class LLMHelper:
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")  # Set your API key in your environment

    def generate_travel_blurb(self, prompt):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message['content'].strip()
        except Exception as e:
            return f"Error generating LLM output: {e}"
