from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
# from langchain_openai import ChatOpenAI

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
import os
from dotenv import load_dotenv

load_dotenv()

watsonx_url = os.getenv("WATSONX_URL")
watsonx_apikey = os.getenv("WATSONX_APIKEY")
project_id = os.getenv("PROJECT_ID")
# model_id="meta-llama/llama-3-2-90b-vision-instruct",
# model_id2="meta-llama/llama-3-3-70b-instruct",

credentials = Credentials(
    url=watsonx_url,
    api_key=watsonx_apikey
)

parameters_1 = {
    "decoding_method": 'greedy',
    "max_new_tokens": 1500,
    "min_new_tokens": 5,
    "temperature": 0
}

model = ModelInference(
    model_id="meta-llama/llama-3-3-70b-instruct",
    credentials=credentials,
    project_id=project_id,
    params=parameters_1
)

from meeting_assistant_flow.types import (
    MeetingTaskList,
)


@CrewBase
class MeetingAssistantCrew:
    """Meeting Assistant Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    # llm = ChatOpenAI(model="gpt-4")
    llm = model

    

    @agent
    def meeting_analyzer(self) -> Agent:
        return Agent(
            config=self.agents_config["meeting_analyzer"],
            llm=self.llm,
        )

    @task
    def analyze_meeting(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_meeting"],
            output_pydantic=MeetingTaskList,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Meeting Issue Generation Crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
