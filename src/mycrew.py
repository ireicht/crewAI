from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task, callback
from crewai.agents.crew_agent_executor import ToolResult
from crewai.agents.parser import AgentAction, AgentFinish
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults #uv pip install langchain_community duckduckgo-search
from crewai.tools import BaseTool
from datetime import datetime
import os.path
from igi_helper import sanitize_filename

from pydantic import Field, BaseModel as PydanticBaseModel
from typing import Type, Union

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


'''
These definitions are needed to describe the tools parameters 
Tool Name: DuckDuckGo Search
Tool Arguments: {{'query': {{'description': 'The term you want to search for.', 'type': 'str'}}}}
Tool Description: Useful for finding up-to-date information from the web.
'''
class DuckDuckGoSearchSchema(PydanticBaseModel):
    query: Union[str, dict] = Field(description="The term you want to search for.")


class myDuckDuckGoSearchTool(BaseTool):

	name: str = "DuckDuckGo Search"
	description: str = "Useful for finding up-to-date information from the web."
	# Define args_schema as a class-level attribute with Pydantic's Field
	args_schema: Type[PydanticBaseModel] = Field(default=DuckDuckGoSearchSchema)

	def _run(self, query: Union[str, dict]) -> str:
		print(f"\nDDGO-query:{query}\n")
		# Ensure the DuckDuckGoSearchRun is invoked properly.
		duckduckgo_tool = DuckDuckGoSearchResults()
		# check if query has nested queries
		if isinstance(query, str):
			duckduckgo_tool = DuckDuckGoSearchResults()
			response = duckduckgo_tool.invoke(query)
			return response
		elif isinstance(query, dict):
			try:
				normalized_query = {k.lower(): v for k, v in query.items()}
				search_string = normalized_query.get("description", None)  # Now you're sure the key is "description"
			except:
				pass
			if not search_string:
				search_string = " ".join([f"{k}:{v}" for k, v in query.items()])

			
			response = duckduckgo_tool.invoke(search_string)
			print(f"DDG_Response: {response}")
			return response

	def _get_tool(self):
		# Create an instance of the tool when needed
		return myDuckDuckGoSearchTool()



@CrewBase
class Mytestcrewa1():
	"""Mytestcrewa1 crew"""

	# additional info for file name outputs
	ts = datetime.now()

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'


	def my_researcher_stepCallback(self, output):
		print(f"\n Researcher STEP PERFORMED: \nstepCallback:{output} \n")
		if isinstance(output, ToolResult):
			print(f"toolresult_finalAnswer ({output.result_as_answer}):{output.result}")
		elif isinstance(output, AgentAction):
			print(f"action output: \nCLBK_RES_Thought: {output.thought} \nCLBK_RES_Tool: {output.tool}\nCLBK_RES_Tool_input: {output.tool_input}\nCLBK_RES_Text: {output.text}\nCLBK_RES_result: {output.result}")

		elif isinstance(output, AgentFinish):
			print(f"agent finish: \nCLBK_RES_Thought: {output.thought}\nCLBK_RES_Output: {output.output}\nCLBK_RES_Text: {output.text}")
		#if agent finish or agent action or Toolresult

	def my_reporter_stepCallback(self, output: Task):
		print(f"\n Reporter STEP PERFORMED: \nstepCallback:{output} \n")
		if isinstance(output, ToolResult):
			print(f"toolresult_finalAnswer ({output.result_as_answer}):{output.result}")
		elif isinstance(output, AgentAction):
			print(f"action output: \nCLBK_REP_Action_Thought: {output.thought} \nCLBK_REP_Action_Tool: {output.tool}\nCLBK_REP_Tool_input: {output.tool_input}\nCLBK_REP_Action_Text: {output.text}\nCLBK_REP_Action_result: {output.result}")

		elif isinstance(output, AgentFinish):
			print(f"agent finish: \nCLBK_REP_Finish_Thought: {output.thought}\nCLBK_REP_Finish_Output: {output.output}\nCLBK_REP_Finish_Text: {output.text}")
		#if agent finish or agent action or Toolresult
	try:
		myllm_llama3_8b = LLM(api_key="fsdf", model="openai/meta-llama-3.1-8b-instruct",  base_url="http://localhost:1234/v1", temperature=0.7, max_tokens=12000)
		# myllm_llama3_8b = LLM(api_key="fsdf", model="openai/meta-llama-3-8b-instruct",  base_url="http://localhost:1234/v1", temperature=0.7, max_tokens=12000)
		myllm_gemma2 = LLM(api_key="fsdf", model="openai/gemma-2-9b-it-8bit",  base_url="http://localhost:1234/v1", temperature=0.0, max_tokens=8192)
		# myllm_gemma2 = LLM(api_key="fsdf", model="openai/gemma-2-27b",  base_url="http://localhost:1234/v1", temperature=0.7, max_tokens=8192)
		myllm_r1_d_qwen = LLM(api_key="fsdf", model="openai/deepseek-r1-distill-qwen-32b-mlx",  base_url="http://localhost:1234/v1", temperature=0.4, max_tokens=12000)
		myllm_r1_d_llama = LLM(api_key="fsdf", model="openai/deepseek-r1-distill-llama-8b",  base_url="http://localhost:1234/v1", temperature=0.4, max_tokens=12000)
		# myllm_mixtral = LLM(api_key="fsdf", model="openai/mixtral-8x7b-instruct-v0.1",  base_url="http://localhost:1234/v1", temperature=0.1, max_tokens=18000)
		myllm_mixtral = LLM(api_key="fsdf", model="openai/mistral-small-24b-instruct-2501",  base_url="http://localhost:1234/v1", temperature=0.0, max_tokens=18000)
		myllm_commandr = LLM(api_key="fsdf", model="openai/c4ai-command-r-v01@q8_0",  base_url="http://localhost:1234/v1", temperature=0.1, max_tokens=18000)
		
		# llm = LLM(api_key="fsdf", model="openai/deepseek-r1-distill-qwen-32b-mlx",  base_url="http://localhost:1234/v1", temperature=0.7)
		# llm=LLM(model="ollama/llama3.2:latest", base_url="http://localhost:11434")
		
	except Exception as e:
		print(f"--> LLM init ERROR {e}")
		
	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	

	@agent
	def prompt_engineer(self) -> Agent:
		return Agent(
			config=self.agents_config['prompt_engineer'],
			verbose=True,
			# step_callback=self.my_researcher_stepCallback,
			# tools=[myDuckDuckGoSearchTool()],
			# llm=self.myllm_mixtral
			llm=self.myllm_gemma2
		)


	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			verbose=True,
			# step_callback=self.my_researcher_stepCallback,
			tools=[myDuckDuckGoSearchTool()],
			# llm=self.myllm_r1_d_llama
			llm=self.myllm_gemma2
		)

	@agent
	def reporting_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['reporting_analyst'],
			verbose=True,
			# step_callback=self.my_reporter_stepCallback,
			# tools=[myDuckDuckGoSearchTool()],
			llm=self.myllm_mixtral
		)


	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task

	@task
	def formulate_prompt_task(self) -> Task:
		myTask=Task(
			config=self.tasks_config['formulate_prompt_task'],
		)
		myTask.output_file=os.path.join('output',f"{self.generateFileName(myTask)}.md")
		return myTask

	@task
	def research_task(self) -> Task:
		myTask=Task(
			config=self.tasks_config['research_task'],
		)
		myTask.output_file=os.path.join('output',f"{self.generateFileName(myTask)}.md")
		return myTask

	@task
	def reporting_task(self) -> Task:
		myTask = Task(
			config=self.tasks_config['reporting_task'],
		)
		myTask.output_file=os.path.join('output',f"{self.generateFileName(myTask)}.md")
		return myTask

	@crew
	def crew(self) -> Crew:
		"""Creates the Mytestcrewa1 crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)

	def generateFileName(self, myTask:Task):	
		return sanitize_filename(f'task_{myTask.name}-model_{myTask.agent.llm.model}_temp_{myTask.agent.llm.temperature}_ts_{self.ts}')