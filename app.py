import os
from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from openai import OpenAI


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


# ==========================================
# APP
# ==========================================

app = FastAPI(title="H&Z Event Organizer")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# DATA MODELS
# ==========================================

class EventRequest(BaseModel):
    event_name: str
    event_type: str
    date: str
    guests: int
    location: str
    budget: float
    venue_preference: str = "Not sure"


class ChatRequest(BaseModel):
    message: str
    event_context: str = ""


# ==========================================
# OPENROUTER CLIENT
# ==========================================

def get_client():

    if not OPENROUTER_API_KEY:
        raise Exception(
            "OPENROUTER_API_KEY was not found in your .env file."
        )

    return OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )


# ==========================================
# HOME / TEST
# ==========================================

@app.get("/")
def home():

    return {
        "success": True,
        "message": "H&Z Event Organizer is running!"
    }


# ==========================================
# CREATE EVENT PLAN
# ==========================================

@app.post("/create-event")
def create_event(event: EventRequest):

    try:

        client = get_client()

        prompt = f"""
You are H&Z Event Organizer, a helpful AI event planner.

Create a practical and organized event plan.

EVENT INFORMATION:

Event name: {event.event_name}
Event type: {event.event_type}
Date: {event.date}
Guests: {event.guests}
Location: {event.location}
Budget: PKR {event.budget:,.0f}
Venue preference: {event.venue_preference}

IMPORTANT:

Do NOT make the answer extremely long.

Do NOT create a huge professional report.

Give the answer in a clean, easy-to-read format.

Do NOT include a long "event concept" section.

Do NOT include a separate "theme" section unless the user specifically asks for one.

Use PKR.

Include:

1. Quick event summary
2. Recommended schedule
3. Food and drinks
4. Decorations
5. Entertainment / activities
6. Budget breakdown in PKR
7. Important tasks

If the venue preference is "Not sure", briefly recommend whether Home,
Hotel/Event Venue, or Outdoor would be most suitable.

Keep the total budget realistic.
"""

        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are H&Z Event Organizer. "
                        "Give practical, concise event-planning help."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response.choices[0].message.content

        return {
            "success": True,
            "ai_plan": answer
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


# ==========================================
# AI CHAT ASSISTANT
# ==========================================

@app.post("/chat")
def chat(request: ChatRequest):

    try:

        client = get_client()

        context = request.event_context

        prompt = f"""
You are H&Z AI Assistant.

You help the user with event planning AND general event-related writing.

You can help with:

- Event planning
- Making an event cheaper
- Food suggestions
- Vegetarian food
- Decoration ideas
- Entertainment
- Venue ideas
- Birthday ideas
- Wedding ideas
- Party ideas
- School events
- Corporate events
- WhatsApp invitations
- Formal invitation letters
- Thank-you letters
- Event announcements
- Speeches
- Messages
- Changing an existing event plan

Current event information, if available:

{context}

User's request:

{request.message}

Answer directly and helpfully.

Keep answers reasonably short unless the user asks for something detailed.
If the user asks you to write something, provide the finished text they can copy.
"""

        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are H&Z AI Assistant. "
                        "You are friendly, practical and helpful."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response.choices[0].message.content

        return {
            "success": True,
            "reply": answer
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


# ==========================================
# VENUE RECOMMENDATIONS
# ==========================================

@app.post("/recommend-venues")
def recommend_venues(event: EventRequest):

    try:

        client = get_client()

        prompt = f"""
You are H&Z Event Organizer.

The user wants venue recommendations.

Location:
{event.location}

Event:
{event.event_type}

Guests:
{event.guests}

Budget:
PKR {event.budget:,.0f}

Venue preference:
{event.venue_preference}

Give useful venue suggestions.

IMPORTANT:

Do NOT pretend that you have live availability.

If you know well-known hotels or venues in the city, mention them,
but clearly say that the user should confirm current prices and availability.

Give a mixture of:

1. Budget-friendly options
2. Mid-range options
3. Hotel options

For each option give:

- Name
- Type
- Why it may suit the event
- Approximate budget category

If you are not confident about a specific real venue, do not invent one.

Keep the answer concise.
"""

        response = client.chat.completions.create(
            model="openrouter/free",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful venue recommendation assistant. "
                        "Never claim live availability unless provided."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response.choices[0].message.content

        return {
            "success": True,
            "venues": answer,
            "real_venues": []
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


# ==========================================
# SERVE FRONTEND
# ==========================================

app.mount(
    "/frontend",
    StaticFiles(
        directory="frontend",
        html=True
    ),
    name="frontend"
)