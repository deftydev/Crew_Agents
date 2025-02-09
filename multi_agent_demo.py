import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_groq import ChatGroq

load_dotenv()

SERPER_API = os.getenv("SERPER_API_KEY")
GROQ_API = os.getenv("GROQ_API_KEY")

search_tool = SerperDevTool()

# Function to create a research agent with memory and delegation control
def create_research_agent(role, task_description, allow_delegation=True):
    llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key=GROQ_API)
    
    return Agent(
        role=role,
        goal=task_description,
        backstory="You are an experienced researcher with expertise in finding and synthesizing information from various sources.",
        verbose=True,
        allow_delegation=allow_delegation,  # Set delegation flag based on the agent's role
        tools=[search_tool],
        llm=llm,
        memory=True  # Pass memory here
    )

# Function to create a task for an agent
def create_research_task(agent, task_description):
    return Task(
        description=task_description,
        agent=agent,
        expected_output="A detailed summary of the research findings, including key points and insights related to the topic."
    )

# Create the agents with specific roles and tasks, ensuring delegation control for the second agent
def create_agents_and_tasks(topic):
   
    agent1_task = f"Conduct an in-depth search on the topic: {topic} and gather relevant sources."
    agent2_task = f"Summarize and Analyze the gathered information for the topic: {topic}."

    # Create agents with memory and delegation settings
    agent1 = create_research_agent(role="Research Gatherer", task_description=agent1_task, allow_delegation=True)
    agent2 = create_research_agent(role="Information Synthesizer", task_description=agent2_task, allow_delegation=False)  # No delegation

    # Create tasks for each agent
    task1 = create_research_task(agent1, agent1_task)
    task2 = create_research_task(agent2, agent2_task)

    # Return the agents and tasks as a tuple
    return [agent1, agent2], [task1, task2]

# Function to run research with multiple agents and memory
def run_research_with_multiple_agents_and_memory(topic):
    agents, tasks = create_agents_and_tasks(topic)
    crew = Crew(agents=agents, tasks=tasks)  # Passing a list of agents and tasks to Crew
    result = crew.kickoff()
    return result

if __name__ == "__main__":
    print("Welcome to the Multi-Agent Research System with Memory!")
    topic = input("Enter the research topic: ")
    result = run_research_with_multiple_agents_and_memory(topic)
    print("Research Result:")
    print(result)
