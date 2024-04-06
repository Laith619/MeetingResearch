import discord
from discord.ext import commands
import os
import httpx

# Initialize Discord Bot
bot = commands.Bot(command_prefix="!")  # You can set your own prefix

DISCORD_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
FASTAPI_ENDPOINT = 'https://meetingresearch.onrender.com/prepare_meeting/'

@bot.event
async def on_ready():
    print(f'{bot.user.name} has connected to Discord!')

@bot.command(name='prepare_meeting', help='Prepares a meeting with the given details')
async def prepare_meeting(ctx, participants: str, context: str, objective: str):
    # Construct the payload to send to the FastAPI endpoint
    payload = {
        'participants': participants,
        'context': context,
        'objective': objective
    }

    # Send a request to the FastAPI application
    try:
        response = httpx.post(FASTAPI_ENDPOINT, json=payload)
        response.raise_for_status()
        result = response.json()
        await ctx.send(result.get("message", "Meeting preparation complete."))
    except httpx.HTTPError as err:
        await ctx.send(f'HTTP error occurred: {err}')
    except Exception as err:
        await ctx.send(f'An error occurred: {err}')

# Run the bot
bot.run(DISCORD_TOKEN)
