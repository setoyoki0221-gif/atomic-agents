from typing import Callable, List
from pydantic import BaseModel, Field
from datetime import datetime
from abc import ABC, abstractmethod
from atomic_agents.lib.chat_memory import ChatMemory

class BasicChatAgentInputSchema(BaseModel):
    chat_input: str = Field(..., description='The input text for the chat agent.')

class BasicChatAgentResponse(BaseModel):    
    response: str = Field(..., description='The markdown-enabled response from the chat agent.')
    
    class Config:
        title = 'BasicChatAgentResponse'
        description = 'Response from the basic chat agent. The response can be in markdown format.'
        json_schema_extra = {
            'title': title,
            'description': description
        }

class SystemPrompt(BaseModel):
    background: List[str]
    steps: List[str]
    output_instructions: List[str]

class DynamicInfoProviderBase(ABC):
    def __init__(self, title: str):
        self.title = title

    @abstractmethod
    def get_info(self) -> str:
        pass

class SystemPromptGenerator:
    def __init__(self, system_prompt: SystemPrompt, dynamic_info_providers: List[DynamicInfoProviderBase] = None):
        self.system_prompt = system_prompt
        self.dynamic_info_providers = dynamic_info_providers or []

    def generate_prompt(self) -> str:
        system_prompt = ''

        if self.system_prompt.background:
            system_prompt += '# IDENTITY and PURPOSE\n'
            system_prompt += f'-{'\n-'.join(self.system_prompt.background)}\n\n'

        if self.system_prompt.steps:
            system_prompt += '# INTERNAL ASSISTANT STEPS\n'
            system_prompt += f'-{'\n-'.join(self.system_prompt.steps)}\n\n'

        if self.system_prompt.output_instructions:
            system_prompt += '# OUTPUT INSTRUCTIONS\n'
            system_prompt += f'-{'\n-'.join(self.system_prompt.output_instructions)}\n\n'

        if self.dynamic_info_providers:
            system_prompt += '# EXTRA INFORMATION AND CONTEXT\n'
            for provider in self.dynamic_info_providers:
                system_prompt += f'## {provider.title}\n'
                system_prompt += f'{provider.get_info()}\n\n'

        return system_prompt

class BasicChatAgent:
    def __init__(self, client, model: str = 'gpt-3.5-turbo', system_prompt: SystemPrompt = None, memory: ChatMemory = None, vector_db = None, input_schema = BasicChatAgentInputSchema, output_schema = BasicChatAgentResponse, dynamic_info_providers: List[DynamicInfoProviderBase] = None):
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.client = client
        self.model = model
        self.memory = memory or ChatMemory()
        self.system_prompt = system_prompt or SystemPrompt(
            background=['You are a helpful AI assistant.'],
            steps=[],
            output_instructions=[]
        )
        self.prompt_generator = SystemPromptGenerator(self.system_prompt, dynamic_info_providers)

    def get_system_prompt(self) -> str:
        return self.prompt_generator.generate_prompt()

    def get_response(self, response_model=None) -> BaseModel:
        if response_model is None:
            response_model = self.output_schema
        
        messages = [{'role': 'system', 'content': self.get_system_prompt()}] + self.memory.get_history()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            response_model=response_model
        )
        return response

    def run(self, user_input: str) -> str:
        self.memory.add_message('user', user_input)
        response = self.get_response(response_model=self.output_schema)
        self.memory.add_message('assistant', response.model_dump_json())
        return response

if __name__ == '__main__':
    import os
    import instructor
    import openai
    from groq import Groq
    from rich.console import Console

    class CurrentDateProvider(DynamicInfoProviderBase):
        def __init__(self, title: str, format: str = '%Y-%m-%d %H:%M:%S'):
            super().__init__(title)
            self.format = format

        def get_info(self) -> str:
            return f'The current date, in the format "{self.format}", is {datetime.now().strftime(self.format)}'

    class LiveSearchResultsProvider(DynamicInfoProviderBase):
        def __init__(self, title: str, search_query: str):
            super().__init__(title)
            self.search_query = search_query

        def get_info(self) -> str:
            # Placeholder for live search results fetching logic
            # In a real implementation, this would fetch live data from a search API
            return f'Live search results for "{self.search_query}"'
    
    console = Console()
    
    # Configuration dictionary
    config = {
        'openai': {
            'client': instructor.from_openai(openai.OpenAI()),
            'model': 'gpt-3.5-turbo'
        },
        'groq': {
            'client': instructor.from_groq(Groq(api_key=os.getenv('GROQ_API_KEY'))),
            'model': os.getenv('GROQ_CHAT_MODEL')
        }
    }
    
    # Choose the configuration
    use_groq = False  # Set this to True to use Groq, False to use OpenAI
    chosen_config = config['groq'] if use_groq else config['openai']
    
    class PlanStep(BaseModel):
        step: str
        description: str
        substeps: List[str] = []
    
    class ArticleBuddyPlanResponse(BaseModel):
        observations: List[str] = Field(..., description='What is the conversation about? What are the key points?')
        thoughts: List[str] = Field(..., description='What is the thought process involved in preparing the ideas and outline?')
        response_plan: List[PlanStep] = Field(..., description='What are the steps involved in generating the ideas and outline?')
        
        class Config:
            title = 'ArticleBuddyPlanResponse'
            description = 'Response plan from the Article Buddy chat agent.'
            json_schema_extra = {
                'title': title,
                'description': description
            }

    class ArticleBuddyResponse(BaseModel):
        response: str = Field(..., description='The markdown-enabled response from the chat agent.')
        
        class Config:
            title = 'ArticleBuddyResponse'
            description = 'Response from the Article Buddy chat agent.'
            json_schema_extra = {
                'title': title,
                'description': description
            }

    class MyChatAgent(BasicChatAgent):
        def run(self, user_input: str) -> ArticleBuddyResponse:
            self.memory.add_message('user', user_input)
            self.memory.add_message('assistant', 'First, let me note the observations about the input and thought process involved in preparing the ideas and outline.')
            plan = self.get_response(response_model=ArticleBuddyPlanResponse)
            self.memory.add_message('assistant', plan.model_dump_json())
            response = self.get_response(response_model=ArticleBuddyResponse)
            self.memory.add_message('assistant', response.model_dump_json())
            return response

    system_prompt = SystemPrompt(
        background=[
            'This assistant is an expert in brainstorming ideas and creating article outlines.',
            'This assistant\'s goal is to generate engaging and creative ideas and structured outlines for articles based on the given topic.'
        ],
        steps=[
            'Note the observations about the input and thought process involved in preparing the ideas and outline.',
            'Respond with a list of 10 ideas and a structured outline for the article in a markdown-formatted text.'
        ],
        output_instructions=[
            'Ensure each idea is clearly numbered from 1 to 10.',
            'Output the list of ideas and the article outline in human-readable Markdown format.',
            'Do not include any additional text or confirmation messages; only provide the list of ideas and the article outline.',
            'Do not reply to the user\'s input with anything except the list of ideas and the article outline.'
        ]
    )

    memory = ChatMemory()
    initial_memory = [
        {'role': 'assistant', 'content': 'Hello! I\'m an AI assistant that can help you brainstorm ideas and generate article outlines. Please provide me with a topic to get started.'}
    ]
    memory.load_from_dict_list(initial_memory)

    # Define dynamic info providers
    dynamic_info_providers = [
        CurrentDateProvider('Current date', format='%Y-%m-%d %H:%M:%S'),
    ]

    agent = MyChatAgent(
        client=chosen_config['client'], 
        model=chosen_config['model'],
        system_prompt=system_prompt,
        memory=memory,
        dynamic_info_providers=dynamic_info_providers,
        output_schema=ArticleBuddyResponse
    )
    console.print(f'Agent: {initial_memory[0]["content"]}')

    while True:
        user_input = input('You: ')
        if user_input.lower() in ['exit', 'quit']:
            print('Exiting chat...')
            break

        response = agent.run(user_input)
        console.print(f'Agent: {response.response}')
