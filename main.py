from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import json 
from typing import Optional
import re
from json.decoder import JSONDecodeError
import logging



from crewai import Crew

from tasks import MeetingPreparationTasks
from agents import MeetingPreparationAgents

# Setup logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")


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
    # Log the request headers for debugging
    headers = request.headers
    logger.info(f"Request headers: {headers}")
    
    # Retrieve the body as bytes and then log it
    body_bytes = await request.body()
    body_text = body_bytes.decode('utf-8')  # Decode bytes to string
    logger.info(f"Request body: {body_text}")

    # Check the Content-Type of the incoming request is 'application/json'
    if 'application/json' not in headers.get('Content-Type', ''):
        logger.error(f"Received Content-Type: {headers.get('Content-Type')}")
        raise HTTPException(status_code=415, detail="Unsupported Media Type. Please send JSON.")
    
    try:
        # Parse the body text into JSON
        data = json.loads(body_text)
        logger.info(f"Parsed JSON body: {data}")
        
        # Validate data against the Pydantic model
        meeting_request = MeetingRequest(**data)
        logger.info(f"Validated meeting request data: {meeting_request}")

        # Extract participant emails, context, and objective from the request
        participant_emails = meeting_request.participants.split(',')
        context = meeting_request.context
        objective = meeting_request.objective

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

        # Initialize the crew and kickoff the meeting preparation
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
    
    except JSONDecodeError as e:
        logger.error(f"JSONDecodeError: {e}")
        raise HTTPException(status_code=400, detail="Malformed JSON or empty payload.")
    except Exception as e:
        logger.exception("Error during the preparation of the meeting")
        raise HTTPException(status_code=500, detail="Internal server error during meeting preparation.")
