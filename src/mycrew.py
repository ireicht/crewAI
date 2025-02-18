from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task, callback
from crewai.agents.crew_agent_executor import ToolResult
from crewai.agents.parser import AgentAction, AgentFinish

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class Mytestcrewa1():
	"""Mytestcrewa1 crew"""

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
		myllm_llama3_8b = LLM(api_key="fsdf", model="openai/meta-llama-3.1-8b-instruct",  base_url="http://localhost:1234/v1", temperature=0.1)
		# myllm_gemma2 = LLM(api_key="fsdf", model="openai/gemma-2-9b-it-8bit",  base_url="http://localhost:1234/v1", temperature=0.7, max_tokens=0)
		myllm_gemma2 = LLM(api_key="fsdf", model="openai/gemma-2-27b",  base_url="http://localhost:1234/v1", temperature=0.7, max_tokens=0)
		myllm_r1_d_qwen = LLM(api_key="fsdf", model="openai/deepseek-r1-distill-qwen-32b-mlx",  base_url="http://localhost:1234/v1", temperature=0.7)
		
		# llm = LLM(api_key="fsdf", model="openai/deepseek-r1-distill-qwen-32b-mlx",  base_url="http://localhost:1234/v1", temperature=0.7)
		# llm=LLM(model="ollama/llama3.2:latest", base_url="http://localhost:11434")
		
	except Exception as e:
		print(f"--> LLM init ERROR {e}")
		
	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def researcher(self) -> Agent:
		return Agent(
			config=self.agents_config['researcher'],
			verbose=True,
			step_callback=self.my_researcher_stepCallback,
			llm=self.myllm_llama3_8b
		)

	@agent
	def reporting_analyst(self) -> Agent:
		return Agent(
			config=self.agents_config['reporting_analyst'],
			verbose=True,
			step_callback=self.my_reporter_stepCallback,
			llm=self.myllm_llama3_8b
		)

	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task
	@task
	def research_task(self) -> Task:
		return Task(
			config=self.tasks_config['research_task'],
		)

	@task
	def reporting_task(self) -> Task:
		return Task(
			config=self.tasks_config['reporting_task'],
			output_file='report.md'
		)

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
