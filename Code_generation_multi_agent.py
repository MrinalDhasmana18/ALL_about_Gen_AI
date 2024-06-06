# %%writefile my_app.py
# How to use CREWAI to implemetn a multi agent framework

# Import necessary modules
from langchain_community.llms import Ollama
import streamlit as st
import os
from crewai import Agent, Task, Crew, Process

# Set up the Streamlit framework
# st.title('GSPANN Python Code Generator App')  # Set the title of the Streamlit app

# Initialize the Ollama model
models = ("llama2","mistral")

# Set a default model for the selectbox
# default_model = "mistral"
# selected_model = st.selectbox("Select a model:", models, index=models.index(default_model))
# selected_model = st.selectbox("Select amodel:", models)

# if st.checkbox("Confirm Selection"):
#     st.write(f"You selected: {selected_model}")
# st.markdown("---")

llm=Ollama(model='llama2:latest')

# input_text = st.text_input("Enter your question here", value="As a developer, I will be creating a correct python code.")
# UserStory=input_text
#UserStory="Create a set of test cases for the user Story:as a user, i want to be to login to my website using a user id and password"


Coder = Agent(
    role='You are a expert software python programmer. You need to develop python code',
    goal='''As a programmer, you are required to complete the function. Use a Chain-of-Thought approach to break
down the problem, create pseudocode, and then write the code in Python language. Ensure that your code is
efficient, readable, and well-commented.
Instructions:
       Understand and Clarify: Make sure you understand the task.
       Algorithm/Method Selection: Decide on the most efficient way.
       Pseudocode Creation: Write down the steps you will follow in pseudocode.
       Code Generation: Translate your pseudocode into executable Python code''',
    
    backstory='Your job is to understand the Question and generate the workable python code for the question asked.',
    llm=llm,
    verbose=True,
    allow_delegation=False
)



Tester  = Agent(    
    
    role='As a tester, your task is to create Basic and Simple test cases based on provided Requirement and Python Code.',
    goal='''These test cases should encompass Basic, Edge scenarios to ensure the code's robustness, reliability, and scalability.
**1. Basic Test Cases**:
- **Objective**: Basic and Small scale test cases to validate basic functioning 
**2. Edge Test Cases**:
- **Objective**: To evaluate the function's behavior under extreme or unusual conditions.

   **Instructions**:
- Implement a comprehensive set of test cases based on requirements.
- Pay special attention to edge cases as they often reveal hidden bugs.
- Only Generate Basics and Edge cases which are small
- Avoid generating Large scale and Medium scale test case. Focus only small, basic test-cases''',
    backstory="You will be testing the code generate by coder",
    llm=llm,
    verbose=True,
    allow_delegation=False
)



Reviewer = Agent(
    role = '''You have to add testing layer in the *Python Code* that can help to execute the code. 
    You need to pass only Provided Input as argument and validate if the Given Expected Output is matched.''',
    goal = '''- Make sure to return the error if the assertion fails
    - Generate the code that can be execute''',
    backstory = 'Code we have generated must be executable.',
    llm=llm,
    verbose=True,
    allow_delegation=False
)


task1 = Task(
  description="""Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.

The overall run time complexity should be O(log (m+n)).""",
  expected_output='Python code which is computationally excellent',
  agent=Coder
    
  )
task2 = Task(
  description='generating test cases for the code generated ',
  expected_output='Test cases generated to test the code',
  agent=Tester
  )

task3 = Task(description = 'You will review the tested code and check if it is executable checking all the updated libraries and requirements as well',
             expected_output="Print the python code and refined code",
             agent = Reviewer
             
)


crew = Crew(
    agents = [Coder,Tester,Reviewer],
    tasks = [task1,task2,task3],
    verbose=2,
    process=Process.sequential
)

# create a streamlit action button to kickoff the process
# if st.button("Generate Python Code"):
#     with st.spinner('Generating Code... Please wait'):
result = crew.kickoff()
print(result)


