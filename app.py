import os
import streamlit as st
from crewai import Agent, Crew, Process, Task
from crewai_tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq

# Streamlit Page Config
st.set_page_config(page_title="Multi-Agent Research Assistant", page_icon="🔬", layout="wide")

st.title("🔬 Autonomous Multi-Agent Research Team")
st.write("Powered by **CrewAI**, **Groq LLM**, and **DuckDuckGo Search**.")

# Sidebar - API Key and Settings
with st.sidebar:
    st.header("Settings")
    groq_api_key = st.text_input("Enter Groq API Key:", type="password")
    
    model_choice = st.selectbox(
        "Select Groq Model:",
        options=["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
        index=0
    )
    st.info("GPT-OSS 120B is recommended for complex analytical tasks.")

research_topic = st.text_input(
    "Research Topic:", 
    placeholder="e.g., Impact of Quantum Computing on Financial Cybersecurity in 2026"
)

# Start Research Process
if st.button("Start Research", type="primary"):
    if not groq_api_key:
        st.error("Please enter your Groq API Key in the sidebar.")
    elif not research_topic.strip():
        st.warning("Please enter a valid research topic.")
    else:
        try:
            # Set up Groq LLM via LangChain Groq provider
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name=model_choice,
                temperature=0.3
            )

            # Online Web Search Tool (No local installations required)
            search_tool = DuckDuckGoSearchRun()

            # --- AGENT DEFINITIONS ---
            researcher = Agent(
                role="Senior Web Researcher",
                goal=f"Gather accurate, detailed, and up-to-date research data on: {research_topic}",
                backstory="An expert researcher who excels at discovering key insights, statistics, and verifiable sources online.",
                tools=[search_tool],
                llm=llm,
                verbose=True,
                allow_delegation=False
            )

            analyst = Agent(
                role="Data Analyst & Writer",
                goal="Synthesize research data into structured, easy-to-read sections.",
                backstory="An skilled technical writer capable of converting raw web findings into organized, coherent drafts.",
                llm=llm,
                verbose=True,
                allow_delegation=False
            )

            editor = Agent(
                role="Fact-Checker & Quality Editor",
                goal="Review, refine, and format the final document for publication clarity.",
                backstory="A meticulous chief editor who ensures formatting consistency, removes redundancies, and refines prose.",
                llm=llm,
                verbose=True,
                allow_delegation=False
            )

            # --- TASK DEFINITIONS ---
            task_research = Task(
                description=f"Search the web and gather primary facts, findings, and recent data on '{research_topic}'.",
                expected_output="Detailed raw research findings categorized by sub-topics with source references.",
                agent=researcher
            )

            task_analysis = Task(
                description="Organize the raw research data into a clear report outline with key takeaways, trends, and evidence.",
                expected_output="A well-structured initial draft with clear headings, analysis, and key observations.",
                agent=analyst
            )

            task_edit = Task(
                description="Review the initial draft. Polish language, ensure objective tone, format nicely with Markdown, and add a brief Summary section.",
                expected_output="A publication-ready final research report in Markdown format.",
                agent=editor
            )

            # --- CREW SETUP ---
            research_crew = Crew(
                agents=[researcher, analyst, editor],
                tasks=[task_research, task_analysis, task_edit],
                process=Process.sequential,
                verbose=True
            )

            with st.spinner("Agent team is conducting research... (This takes 1-2 minutes)"):
                result = research_crew.kickoff()

            st.success("Research Completed!")
            st.markdown("---")
            st.markdown(result)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
