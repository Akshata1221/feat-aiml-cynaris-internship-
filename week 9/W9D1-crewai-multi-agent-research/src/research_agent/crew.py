"""CrewAI implementation: Researcher -> Writer -> Reviewer."""
from .config import settings
from .search import search_web

def build_crew(topic: str):
    try:
        from crewai import Agent, Task, Crew, Process
        from crewai.tools import BaseTool
        from pydantic import Field
    except ImportError as exc:
        raise RuntimeError("Install project dependencies to use CrewAI mode.") from exc

    class WebResearchTool(BaseTool):
        name: str = "Web research search"
        description: str = "Search the web for recent information and return titles, URLs and snippets."
        query: str = Field(default="")

        def _run(self, query: str) -> str:
            return "\n".join(
                f"Title: {s.title}\nURL: {s.url}\nSnippet: {s.snippet}"
                for s in search_web(query)
            )

    researcher = Agent(
        role="Research Analyst",
        goal="Find relevant, credible and recent evidence for the given research topic.",
        backstory="You are a careful analyst who prioritizes primary sources, dates and traceable evidence.",
        tools=[WebResearchTool()], verbose=True, allow_delegation=False,
    )
    writer = Agent(
        role="Technical Report Writer",
        goal="Turn verified research notes into a clear, structured and balanced report.",
        backstory="You write concise reports, distinguish evidence from opinion and cite every factual claim.",
        verbose=True, allow_delegation=False,
    )
    reviewer = Agent(
        role="Quality Reviewer",
        goal="Audit the report for unsupported claims, missing citations, bias and clarity.",
        backstory="You are a rigorous editor who prefers transparent limitations over invented certainty.",
        verbose=True, allow_delegation=False,
    )
    research_task = Task(
        description=f"Research the topic: {topic}. Use the web research tool. Return 5-8 concise findings with source title, URL, date if available, and evidence. Flag uncertainty.",
        expected_output="Structured research notes with traceable URLs and caveats.",
        agent=researcher,
    )
    writing_task = Task(
        description="Write an executive summary, key findings, implications, limitations and a source list from the research notes. Do not invent facts or citations.",
        expected_output="A Markdown research report with clear sections and inline source links.",
        agent=writer,
        context=[research_task],
    )
    review_task = Task(
        description="Review the report. Identify unsupported claims, citation gaps and unclear wording. Return a corrected final Markdown report plus a short review checklist.",
        expected_output="Reviewed Markdown report with source links, limitations and reviewer notes.",
        agent=reviewer,
        context=[research_task, writing_task],
    )
    return Crew(agents=[researcher, writer, reviewer],
                tasks=[research_task, writing_task, review_task],
                process=Process.sequential, verbose=True)

def run_crew(topic: str) -> str:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for CrewAI mode. Use --mode offline otherwise.")
    crew = build_crew(topic)
    return str(crew.kickoff())
