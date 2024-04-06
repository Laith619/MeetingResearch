from fastapi import FastAPI, HTTPException, Request
import re
from typing import Optional

# Assuming 'crewai', 'tasks', and 'agents' are your custom modules.
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
async def prepare_meeting(request: Request):
    # Get the raw text body from the request
    body = await request.json()
    message = body.get("message", "")

    # Define the regex pattern for parsing the SMS text
    pattern = r"participants: ([^;]+); context: ([^;]+); objective: (.+)"
    match = re.match(pattern, message)

    # If the pattern doesn't match, return an error
    if not match:
        raise HTTPException(status_code=400, detail="Invalid message format.")

    # Extract matched groups into variables
    participants, context, objective = match.groups()

    # Split participants by comma to create a list
    participant_emails = participants.split(',')

    # Initialize agents and tasks (assuming you have these functions defined)
    researcher_agent = agents.research_agent()
    industry_analyst_agent = agents.industry_analysis_agent()
    meeting_strategy_agent = agents.meeting_strategy_agent()
    summary_and_briefing_agent = agents.summary_and_briefing_agent()

    # Execute tasks (assuming these functions are also defined in your modules)
    research = tasks.research_task(researcher_agent, participant_emails, context)
    industry_analysis = tasks.industry_analysis_task(industry_analyst_agent, participant_emails, context)
    meeting_strategy = tasks.meeting_strategy_task(meeting_strategy_agent, context, objective)
    summary_and_briefing = tasks.summary_and_briefing_task(summary_and_briefing_agent, context, objective)

    # Create a Crew instance with the agents and tasks
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

    # Execute the tasks
    result = crew.kickoff()

    # Return the result as JSON
    return {
        "result": result,
        "message": "Meeting prepared successfully!"
    }
