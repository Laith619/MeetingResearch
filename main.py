from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import json
import re
import logging
from json.decoder import JSONDecodeError
from typing import Optional


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
    try:
        content = await request.json()
        message = content.get('message', '')

        def parse_message(msg: str) -> dict:
            pattern = r'participants: ([^;]+); context: ([^;]+); objective: (.+)'
            match = re.match(pattern, msg)
            if not match:
                raise ValueError('Message format is incorrect.')
            return {
                'participants': match.group(1),
                'context': match.group(2),
                'objective': match.group(3)
            }

        parsed_data = parse_message(message)
        meeting_request = MeetingRequest(**parsed_data)
        logger.info(f'Validated meeting request data: {meeting_request}')

        participant_emails = meeting_request.participants.split(',')
        context = meeting_request.context
        objective = meeting_request.objective

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

        return {
            "result": result,
            "message": "Meeting prepared successfully!"
        }

    except JSONDecodeError as e:
        logger.error(f'JSONDecodeError: {e}')
        raise HTTPException(status_code=400, detail='Malformed JSON or empty payload.')
    except ValueError as e:
        logger.error(f'Parsing error: {e}')
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f'Unexpected error: {e}')
        raise HTTPException(status_code=500, detail='Internal server error.')
