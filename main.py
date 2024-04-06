from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from twilio.rest import Client
import json
import httpx
import re
import logging
import os
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


# Twilio setup
account_sid = os.getenv('TWILIO_ACCOUNT_SID')  # Set this in your environment variables
auth_token = os.getenv('TWILIO_AUTH_TOKEN')    # Set this in your environment variables
twilio_number = os.getenv('TWILIO_PHONE_NUMBER')  # Your Twilio phone number


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


        # Send an SMS using Twilio
        try:
            # Initialize the Twilio client
            client = Client(account_sid, auth_token)
            
            # Assuming the recipient's phone number is passed in the request or set as an environment variable
            to_number = os.getenv('RECIPIENT_PHONE_NUMBER') # Or use a different method to retrieve this number
            
            # Send the SMS message
            message = client.messages.create(
                body="Your meeting has been prepared successfully.",
                from_=twilio_number,
                to=to_number
            )
            logger.info(f"Message sent: {message.sid}")

        except Exception as e:
            logger.error(f"Failed to send SMS: {e}")
            response_message = "Meeting prepared, but failed to send SMS notification."

    except JSONDecodeError as e:
        logger.error(f'JSONDecodeError: {e}')
        raise HTTPException(status_code=400, detail='Malformed JSON or empty payload.')
    except ValueError as e:
        logger.error(f'Parsing error: {e}')
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f'Unexpected error: {e}')
        raise HTTPException(status_code=500, detail='Internal server error.')
    
    # Return the final response
    return {"message": response_message}
