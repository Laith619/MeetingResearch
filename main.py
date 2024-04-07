from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import json
import httpx
import re
import logging
import os
from json.decoder import JSONDecodeError

from crewai import Crew
from tasks import MeetingPreparationTasks
from agents import MeetingPreparationAgents

# Setup logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

# Define the request model
class MeetingRequest(BaseModel):
    participants: str
    context: str
    objective: str

# Discord Webhook URL from environment variables
DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL')

# Initialize FastAPI app and components
app = FastAPI()
tasks = MeetingPreparationTasks()
agents = MeetingPreparationAgents()

@app.post("/prepare_meeting/")
async def prepare_meeting(request: Request):
    try:
        raw_body = await request.body()
        logger.info(f"Raw request body: {raw_body}")
        content = await request.json()

        # Initialize variables for participants, context, and objective
        participants = context = objective = None

        # Check if 'message' is a list (new format) and handle accordingly
        if isinstance(content.get('message'), list):
            # Extract data from the first item in the list
            message_data = content['message'][0]
            participants = message_data.get('participants')
            context = message_data.get('context')
            objective = message_data.get('objective')
        elif isinstance(content.get('message'), str):
            # Existing code for handling string message (old format)
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
            participants = parsed_data['participants']
            context = parsed_data['context']
            objective = parsed_data['objective']

        # Validate the extracted data
        if not all([participants, context, objective]):
            raise ValueError('One or more required fields are missing.')

        meeting_request = MeetingRequest(
            participants=participants,
            context=context,
            objective=objective
        )
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

        crew = Crew(agents=[researcher_agent, industry_analyst_agent, meeting_strategy_agent, summary_and_briefing_agent], 
                    tasks=[research, industry_analysis, meeting_strategy, summary_and_briefing])
        

        result = crew.kickoff()
        logger.info(f"Meeting preparation result: {result}")

        # Construct an embed message for Discord
        discord_embed = {
            "embeds": [{
                "title": "Meeting Preparation Result",
                "description": f"**Participants:** {meeting_request.participants}\n**Context:** {meeting_request.context}\n**Objective:** {meeting_request.objective}\n**Result:** {result}",
                "color": 0x00ff00  # Green color
            }]
        }

        # Send the embed message to Discord webhook
        discord_response = httpx.post(DISCORD_WEBHOOK_URL, json=discord_embed)
        discord_message = "Notification sent to Discord."
        if discord_response.status_code != 204:
            logger.error(f"Failed to send message to Discord, status code: {discord_response.status_code}, response: {discord_response.text}")
            discord_message = "Failed to send notification to Discord."

        # Send result to albato.com webhook
        albato_webhook_url = 'https://h.albato.com/wh/38/1lfj6jg/w6A5fbJU9tCl6qN6KJXmFxCjeaAxYTYlybzCqhhq3hc/'  
        albato_response = httpx.post(albato_webhook_url, json={'result': result})
        albato_message = "Data sent to Albato webhook successfully."
        if albato_response.status_code != 200:
            logger.error(f"Failed to send data to Albato webhook, status code: {albato_response.status_code}, response: {albato_response.text}")
            albato_message = "Failed to send data to Albato webhook."

        # Return a combined message indicating the results of both webhook requests
        return {"message": f"{discord_message} {albato_message}"}

    except JSONDecodeError as e:
        logger.error(f'JSONDecodeError: {e}')
        raise HTTPException(status_code=400, detail='Malformed JSON or empty payload.')
    except ValueError as e:
        logger.error(f'Parsing error: {e}')
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f'Unexpected error: {e}')
        raise HTTPException(status_code=500, detail='Internal server error.')
