import os
import re
import json
import streamlit as st
from typing import Type, Any, Dict, Optional
from pydantic import BaseModel, Field, ConfigDict, AliasChoices

# -----------------------------------------------------------------------------
# MONKEY PATCH: LITELLM / GROQ CACHE_BREAKPOINT FIX
# -----------------------------------------------------------------------------
import litellm

_original_litellm_completion = litellm.completion

def _patched_litellm_completion(*args, **kwargs):
    if "messages" in kwargs and isinstance(kwargs["messages"], list):
        cleaned_messages = []
        for msg in kwargs["messages"]:
            if isinstance(msg, dict):
                cleaned_msg = {k: v for k, v in msg.items() if k != "cache_breakpoint"}
                cleaned_messages.append(cleaned_msg)
            else:
                cleaned_messages.append(msg)
        kwargs["messages"] = cleaned_messages
    return _original_litellm_completion(*args, **kwargs)

litellm.completion = _patched_litellm_completion

# -----------------------------------------------------------------------------
# IMPORTS & PAGE CONFIGURATION
# -----------------------------------------------------------------------------
from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import BaseTool
from ddgs import DDGS

st.set_page_config(
    page_title="Insight AI | Research Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Responsive, adaptive CSS for Light & Dark mode support
st.markdown("""
<style>
    .main-header {
        padding: 1.5rem;
        border-radius: 12px;
        background: rgba(59, 130, 246, 0.08);
        border: 1px solid rgba(59, 130, 246, 0.2);
        margin-bottom: 1.5rem;
    }
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    .main-subtitle {
        font-size: 0.95rem;
        opacity: 0.8;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# FLEXIBLE PYDANTIC SCHEMA (HANDLES 'Query', 'query', AND EXTRA PROPS)
# -----------------------------------------------------------------------------
class WebSearchInput(BaseModel):
    model_config = ConfigDict(
        extra="allow",  # Allow any unexpected parameters emitted by GPT-OSS models
        populate_by_name=True
    )
    
    query: str = Field(
        ...,
        description="The search query string to look up on the web.",
        validation_alias=AliasChoices("query", "Query", "keywords", "search_query")
    )

class CustomWebSearchTool(BaseTool):
    name: str = "web_search_tool"
    description: str = "Useful to search the web for latest news, developments, and statistics on any topic."
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, **kwargs: Any) -> str:
        try:
            # Flexible extraction to guarantee string retrieval
            search_term = (
                kwargs.get("query") 
                or kwargs.get("Query") 
                or kwargs.get("keywords") 
                or kwargs.get("search_query")
            )

            if not search_term and kwargs:
                # Fallback: grab the first string value passed in kwargs
                for val in kwargs.values():
                    if isinstance(val, str):
                        search_term = val
                        break

            if not search_term:
                return "Search query was empty. Please provide a valid search string."

            # Handle edge cases where LLM passes a serialized JSON string
            if isinstance(search_term, str) and search_term.strip().startswith("{"):
                try:
                    parsed = json.loads(search_term)
                    search_term = parsed.get("query") or parsed.get("Query") or search_term
                except Exception:
                    pass

            cleaned_query = re.sub(r'[^\w\s]', '', str(search_term)).strip()
            
            results = DDGS().text(keywords=cleaned_query, max_results=5)
            
            # Fallback search if exact term returns nothing
            if not results and len(cleaned_query.split()) > 3:
                shorter_query = " ".join(cleaned_query.split()[:3])
                results = DDGS().text(keywords=shorter_query, max_results=5)
                
            if not results:
                return f"No search results found for query: '{cleaned_query}'. Synthesize findings based on core domain knowledge."
            
            formatted_results = []
            for r in results:
                formatted_results.append(
                    f"Title: {r.get('title')}\n"
                    f"Link: {r.get('href')}\n"
                    f"Snippet: {r.get('body')}\n"
                )
            return "\n---\n".join(formatted_results)
        except Exception as e:
            return f"Search execution notice: {str(e)}. Proceeding with synthesis."

web_search_tool = CustomWebSearchTool()

# -----------------------------------------------------------------------------
# MAIN INTERFACE
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <div class="main-title">⚡ Insight AI Research Studio</div>
    <div class="main-subtitle">Autonomous multi-agent research team powered by <b>CrewAI</b> & <b>Groq LPU Acceleration</b></div>
</div>
""", unsafe_allow_html=True)

# Secrets retrieval
secret_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))

with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    
    if secret_key:
        st.success("🔒 Groq API Key loaded from Streamlit Secrets!")
        groq_api_key = secret_key
    else:
        groq_api_key = st.text_input(
            "Enter Groq API Key:", 
            type="password", 
            help="Get your key from console.groq.com"
        )
    
    model_choice = st.selectbox(
        "Groq Architecture Model:",
        options=["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"],
        index=0,
        help="Select the Groq model for your research team."
    )
    
    st.markdown("---")
    st.markdown("### 👥 Active Research Team")
    st.markdown("• **Senior Web Researcher**")
    st.markdown("• **Data Analyst & Writer**")
    st.markdown("• **Fact-Checker & Editor**")
    
    st.markdown("---")
    st.caption("Environment: Python 3.11 | CrewAI 0.30+")

col1, col2 = st.columns([3, 1])

with col1:
    research_topic = st.text_input(
        "Research Target Topic:",
        placeholder="e.g., Solid-state battery commercialization timelines"
    )

with col2:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    run_button = st.button("🚀 Launch Research", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# WORKFLOW EXECUTION
# -----------------------------------------------------------------------------
if run_button:
    if not groq_api_key:
        st.error("🔑 Groq API Key is missing. Please set GROQ_API_KEY in Streamlit Secrets or enter it manually.")
    elif not research_topic.strip():
        st.warning("⚠️ Please enter a research topic.")
    else:
        try:
            os.environ["GROQ_API_KEY"] = groq_api_key

            llm = LLM(
                model=f"groq/{model_choice}",
                temperature=0.1,
                api_key=groq_api_key
            )

            researcher = Agent(
                role="Senior Web Researcher",
                goal=f"Find recent key statistics and developments regarding: {research_topic}",
                backstory="An expert researcher proficient in searching the web with clean search queries.",
                tools=[web_search_tool],
                llm=llm,
                verbose=True,
                allow_delegation=False
            )

            analyst = Agent(
                role="Data Analyst & Synthesizer",
                goal="Organize raw research findings into clear thematic sections.",
                backstory="A technical writer adept at building coherent report structures from research findings.",
                llm=llm,
                verbose=True,
                allow_delegation=False
            )

            editor = Agent(
                role="Chief Quality Editor",
                goal="Format and polish final research report in Markdown format.",
                backstory="A chief publication editor ensuring clean Markdown layout and actionable insights.",
                llm=llm,
                verbose=True,
                allow_delegation=False
            )

            task_research = Task(
                description=(
                    f"Search the web for recent developments regarding '{research_topic}'. "
                    "Call the tool `web_search_tool` using the argument key `query`."
                ),
                expected_output="Categorized research notes, key data points, and relevant context.",
                agent=researcher
            )

            task_analysis = Task(
                description="Consolidate research findings into a structured report outline with key themes.",
                expected_output="A detailed draft report featuring clear headers and organized insights.",
                agent=analyst
            )

            task_edit = Task(
                description="Review and polish the report into clean Markdown format with an Executive Summary.",
                expected_output="A publication-ready Markdown research report.",
                agent=editor
            )

            research_crew = Crew(
                agents=[researcher, analyst, editor],
                tasks=[task_research, task_analysis, task_edit],
                process=Process.sequential,
                verbose=True
            )

            with st.status("🛸 **Multi-Agent Team Active...**", expanded=True) as status:
                st.write("🔍 **Senior Web Researcher:** Querying online search endpoints...")
                result = research_crew.kickoff()
                status.update(label="✅ **Research Execution Complete!**", state="complete", expanded=False)

            st.markdown("### 📊 Generated Research Report")
            st.markdown(result.raw)
            
            st.download_button(
                label="📥 Download Report (.md)",
                data=str(result.raw),
                file_name="research_report.md",
                mime="text/markdown"
            )

        except Exception as e:
            st.error(f"Execution Error: {str(e)}")
    
