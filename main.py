from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic import BaseModel
from typing import Optional

from crewai import Crew

from tasks import MeetingPreparationTasks
from agents import MeetingPreparationAgents

# Define a simplified request model
class MeetingRequest(BaseModel):
    participants: str  # Text containing participant emails, comma-separated
    context: str       # Text describing the meeting context
    objective: str     # Text describing the meeting objective

# Initialize FastAPI app
app = FastAPI()

# Initialize the tasks and agents
tasks = MeetingPreparationTasks()
agents = MeetingPreparationAgents()

@app.post("/prepare_meeting/")
def prepare_meeting(request: MeetingRequest):
    # Parse participants as a list of emails
    participant_emails = request.participants.split(',')

    # Use the other text fields directly
    context = request.context
    objective = request.objective

    # Initialize agents and tasks (similar to your previous logic)
    researcher_agent = agents.research_agent()
    industry_analyst_agent = agents.industry_analysis_agent()
    meeting_strategy_agent = agents.meeting_strategy_agent()
    summary_and_briefing_agent = agents.summary_and_briefing_agent()

    research = tasks.research_task(researcher_agent, participant_emails, context)
    industry_analysis = tasks.industry_analysis_task(industry_analyst_agent, participant_emails, context)
    meeting_strategy = tasks.meeting_strategy_task(meeting_strategy_agent, context, objective)
    summary_and_briefing = tasks.summary_and_briefing_task(summary_and_briefing_agent, context, objective)

    crew = Crew(
        agents=[
            researcher_agent, 
            industry_analyst_agent,
            meeting_strategy_agent, 
            summary_and_briefing_agent
        ], 
        tasks=[
            research, 
            industry_analysis,
            meeting_strategy,
            summary_and_briefing
        ]
    )

    result = crew.kickoff()

    # Return result as JSON
    return {
        "result": result,
        "message": "Meeting prepared successfully!"
    }
