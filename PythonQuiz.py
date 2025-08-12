# imports

from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
import gradio as gr
import os, sys


# The usual start
load_dotenv(override=True)

# Print the key prefixes to help with any debugging
def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )

def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording interest from {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question} asked that User couldn't answer")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as user didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}




tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]


class PythonQuiz:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required for PythonQuiz.")
        self.openai = OpenAI(api_key=self.api_key,base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({"role": "tool","content": json.dumps(result),"tool_call_id": tool_call.id})
        return results
    
    def system_prompt(self):
        system_prompt = """
        You are the quizmaster for an application called "Simple Python Quiz".

        Your role is to ask only scenario-based Python questions related strictly to Deep Learning and Artificial Intelligence.

        - Each scenario must describe a real-world task or problem in AI/ML, such as image classification, NLP sentiment analysis, reinforcement learning, or model deployment.
        - From each scenario, ask only one technical question at a time. After the user responds, you may continue with the next question based on the same scenario.
        - All questions must be open-ended, application-oriented, and encourage problem-solving and best practices.
        - Do not ask the next question until the user has responded to the current one or indicated they wish to skip it.

        ❗ Do not answer questions or provide hints during the quiz.

        🧠 If the user says they don’t know the answer to a question, use the tool `record_unknown_question` to log that specific question.

        ✅ After the user says they are done with the quiz:
        - Review all of the user's answers.
        - Provide constructive feedback on their strengths and areas to improve based on their responses.
        - Summarize any patterns you noticed (e.g., strong understanding of NLP, needs work on data augmentation).
        - Optionally suggest resources or topics to study further.

        📩 Then, thank them for participating and politely ask for their email address to share personalized feedback or follow-up.
        - Use the tool `record_user_details` to log their email.

        🛑 Do not ask questions outside the Deep Learning/AI domain. Avoid basic Python syntax or unrelated topics.
        """
        return system_prompt

    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            response = self.openai.chat.completions.create(model="gemini-2.5-flash-preview-05-20", messages=messages, tools=tools)
            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content
        

if __name__ == "__main__":
    Quiz = PythonQuiz()
    gr.ChatInterface(Quiz.chat, type="messages").launch()
