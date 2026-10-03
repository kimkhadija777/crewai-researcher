import os
import streamlit as st
from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from duckduckgo_search import DDGS
from langchain_groq import ChatGroq

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & MODERN UI STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Insight AI | Research Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern glassmorphism & styled cards
st.markdown("""
<style>
    /* Global styling */
    .stApp {
        background-color: #0e1117;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Container */
    .main-header {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid #374151;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    
    .main-title {
        color: #f9fafb;
        font-size: 2.25rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-subtitle {
        color: #9ca3af;
        font-size: 1rem;
    }
    
    /* Status Badge */
    .agent-badge {
        display: inline-block;
        background: rgba(59, 130, 246, 0.1);
        color: #60a5fa;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    /* Result Box */
    .report-card {
        background: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 2rem;
        color: #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# DEFINE SAFE, NATIVE CREWAI SEARCH TOOL
# -----------------------------------------------------------------------------
@tool("Web Search Tool")
def web_search_tool(query: str) -> str:
    """Searches the internet for accurate, real-time up-to-date facts, statistics, and information."""
    try:
        results = DDGS().text(keywords=query, max_results=5)
        if not results:
            return "No relevant search results found."
        
        formatted_results = []
        for r in results:
            formatted_results.append(f"Title: {r.get('title')}\nLink: {r.get('href')}\nSnippet: {r.get('body')}\n")
        return "\n---\n".join(formatted_results)
    except Exception as e:
        return f"Error executing search: {str(e)}"


# -----------------------------------------------------------------------------
# MAIN UI INTERFACE
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div class="main-title">⚡ Insight AI Research Studio</div>
    <div class="main-subtitle">Autonomous multi-agent research team powered by <b>CrewAI</b> & <b>Groq LPU Acceleration</b></div>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    
    groq_api_key = st.text_input("Groq API Key:", type="password", help="Get your free key from console.groq.com")
    
    model_choice = st.selectbox(
        "Groq Architecture Model:",
        options=["openai/gpt-oss-120b", "openai/gpt-oss-20b"],
        index=0,
        help="120B model delivers deep analytical output; 20B provides faster execution."
    )
    
    st.markdown("---")
    st.markdown("### 👥 Active Research Agents")
    st.markdown("• **Senior Web Researcher** (DuckDuckGo)")
    st.markdown("• **Data Analyst & Writer** (Synthesis)")
    st.markdown("• **Fact-Checker & Editor** (Formatting)")
    
    st.markdown("---")
    st.caption("Target Environment: Python 3.11 | CrewAI 0.30+")

# Inputs Layout
col1, col2 = st.columns([3, 1])

with col1:
    research_topic = st.text_input(
        "Research Target Topic:",
        placeholder="e.g., Quantum Computing Applications in Financial Risk Management 2026"
    )

with col2:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    run_button = st.button("🚀 Launch Research", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# AGENT WORKFLOW EXECUTION
# -----------------------------------------------------------------------------
if run_button:
    if not groq_api_key:
        st.error("🔑 Groq API Key is missing. Please provide a valid key in the sidebar.")
    elif not research_topic.strip():
        st.warning("⚠️ Please specify a valid research topic before launching agents.")
    else:
        try:
            # Initialize Groq LLM
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name=model_choice,
                temperature=0.2
            )

            # Define Agents
            researcher = Agent(
                role="Senior Web Researcher",
                goal=f"Gather reliable, clear, up-to-date facts on: {research_topic}",
                backstory="An expert analytical researcher specializing in identifying primary data points and current insights across the web.",
                tools=[web_search_tool],
                llm=llm,
                verbose=True,
                allow_delegation=False
            )

            analyst = Agent(
                role="Data Analyst & Synthesizer",
                goal="Structure and synthesize raw research into clear domain sections.",
                backstory="A technical writer who converts unorganized web search data into highly structured, coherent reports.",
                llm=llm,
                verbose=True,
                allow_delegation=False
            )

            editor = Agent(
                role="Chief Quality Editor",
                goal="Polish, format, and audit report clarity for publication.",
                backstory="A chief publication editor ensuring high standards of technical accuracy, flow, and Markdown styling.",
                llm=llm,
                verbose=True,
                allow_delegation=False
            )

            # Define Tasks
            task_research = Task(
                description=f"Conduct internet searches to gather reliable sources, statistics, and main developments regarding: {research_topic}",
                expected_output="Categorized list of primary findings, facts, and relevant source summaries.",
                agent=researcher
            )

            task_analysis = Task(
                description="Consolidate the web research findings into an organized, deep-dive report draft.",
                expected_output="Detailed draft broken into logical headers, key statistics, and analytical observations.",
                agent=analyst
            )

            task_edit = Task(
                description="Review the report draft for clarity, correctness, and clean Markdown structure. Add an Executive Summary.",
                expected_output="A publication-ready Markdown research document.",
                agent=editor
            )

            # Assemble Crew
            research_crew = Crew(
                agents=[researcher, analyst, editor],
                tasks=[task_research, task_analysis, task_edit],
                process=Process.sequential,
                verbose=True
            )

            # UI Progress State
            with st.status("🛸 **Multi-Agent Team Active...**", expanded=True) as status:
                st.write("🔍 **Senior Web Researcher:** Querying live web endpoints...")
                result = research_crew.kickoff()
                status.update(label="✅ **Research Execution Complete!**", state="complete", expanded=False)

            # Display Output Report
            st.markdown("### 📊 Generated Research Report")
            st.markdown(f'<div class="report-card">{result.raw}</div>', unsafe_allow_html=True)
            
            # Download Action
            st.download_button(
                label="📥 Download Report (.md)",
                data=str(result.raw),
                file_name="research_report.md",
                mime="text/markdown"
            )

        except Exception as e:
            st.error(f"Execution Error: {str(e)}")
            
