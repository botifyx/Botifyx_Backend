import requests
from requests.auth import HTTPBasicAuth
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel
import uvicorn
from typing import Optional
import json
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Request, Depends
import requests
import os
import random
import string
from fastapi.responses import RedirectResponse
import re
from dotenv import load_dotenv
import os

# Only load dotenv in development environment
if os.path.isfile(".env"):
    load_dotenv()

JIRA_CLIENT_ID = os.getenv("JIRA_CLIENT_ID")
JIRA_CLIENT_SECRET = os.getenv("JIRA_CLIENT_SECRET")
JIRA_REDIRECT_URI = os.getenv("JIRA_REDIRECT_URI")
JIRA_AUTH_URL = os.getenv("JIRA_AUTH_URL")
JIRA_TOKEN_URL = os.getenv("JIRA_TOKEN_URL")
JIRA_API_SCOPE = os.getenv("JIRA_API_SCOPE")
JIRA_BASE_URL = os.getenv("JIRA_BASE_URL")
PROJECT_KEY = os.getenv("PROJECT_KEY")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_client = OpenAI(api_key=OPENAI_API_KEY)

class UserQuery(BaseModel):
    query: str
    project_id: Optional[str] = None
    email_id : str
    access_token : str
    previous_query: Optional[str] = None
    previous_response: Optional[str] = None

# 🔹 Fetch Project Details from JIRA
def fetch_project_details(project_key: str, user_query: str, email_id: str, access_token: str):
    url = f"{JIRA_BASE_URL}/rest/api/3/project/{project_key}"
    auth = HTTPBasicAuth(email_id, access_token)
    headers = {"Accept": "application/json"}

    response = requests.get(url, headers=headers, auth=auth)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    project_data = response.json()
    prompt = f"""
      You are an **AI Assistant for a Project Manager**, responsible for providing **clear, concise, and insightful project updates** based on JIRA data. Your goal is to **respond directly to queries** instead of asking the user for details.  

    ### Project Details from JIRA:  
    {json.dumps(project_data, indent=2)}

    ### User Query:  
    **{user_query}**

    ### Instructions:  
    - **If the user asks for open issues:**  
    - **Directly list all open issues** in a friendly and clear way, including statuses.  
    - Example: *“Sure! Here’s the list of open issues right now:  
        1. **XYZ-123** (In Progress)  
        2. **ABC-456** (Blocked, priority: High)  
        3. **DEF-789** (To Do)  
        4. **GHI-101** (In Review, priority: Medium).  
        Would you like me to update any of these issues?”*
    - **If the query is about project progress:**  
    - Summarize **overall completion %**, key milestones, and any risks.  
    - Example: *“Your project is 78% complete. Development tasks are almost done, but testing is lagging behind by 2 sprints. Would you like to check pending test cases?”*  

    - **If the query asks for a specific issue/task:**  
    - Find the issue and provide its latest status, assignee, and priority.  
    - If the issue isn’t found, respond like: *“Hmm, I couldn’t find issue ABC-456. Would you like me to list all open issues for you?”*  

    - **If data is missing or unclear:**  
    - Provide **the most relevant available information** rather than asking broad clarifications.  
    - If essential data is unavailable, ask for specifics while still giving partial information.  
    - Example: *“I found 3 critical blockers in your project, but I need an issue ID if you’re looking for a specific one.”*  

    - **Always offer the user the next step**, either asking for an update, clarification, or specific action.
    - **Be concise but informative.** Avoid unnecessary details unless requested.

    - **Keep the response concise, conversational, and action-oriented**—No need for lengthy analysis unless specifically requested.  

    """

    ai_response = openai_client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    return {"message": ai_response.choices[0].message.content}

# 🔹 Update Defect in JIRA
def update_defect_in_jira(defect_id: str, status: str, description: str, priority: str, email_id: str, access_token: str):
    if not defect_id:
        raise HTTPException(status_code=400, detail="Defect ID is required to update a defect.")

    url = f"{JIRA_BASE_URL}/rest/api/3/issue/{defect_id}"
    auth = HTTPBasicAuth(email_id, access_token)
    headers = {"Accept": "application/json", "Content-Type": "application/json"}

    payload = {
        "fields": {
            "description": description or "No description provided.",
            "priority": {"name": priority or "Normal"},
        }
    }

    response = requests.put(url, json=payload, headers=headers, auth=auth)

    if response.status_code == 204:
        return {"message": f"Defect {defect_id} updated successfully!"}
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)

def get_all_projects(email_id : str, access_token: str):
    url = f"{JIRA_BASE_URL}/rest/api/3/project"
    auth = HTTPBasicAuth(email_id, access_token)
    headers = {
    "Accept": "application/json"
    }
    response = requests.request(
    "GET",
    url,
    headers=headers,
    auth=auth
    )
    # response_data = json.dumps(json.loads(response.text), sort_keys=True, indent=4, separators=(",", ": "))
    print("projects",json.dumps(json.loads(response.text), sort_keys=True, indent=4, separators=(",", ": ")))
    return response.json()

# 🔹 Create a Defect in JIRA
def create_defect_in_jira(project_key: str, summary: str, description: str, priority: str, email_id: str, access_token: str):
    url = f"{JIRA_BASE_URL}/rest/api/3/issue"
    auth = HTTPBasicAuth(email_id, access_token)
    headers = {"Accept": "application/json", "Content-Type": "application/json"}

    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": summary,
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {"type": "paragraph", "content": [{"type": "text", "text": description}]}
                ]
            },
            "issuetype": {"name": "Task"}
        }
    }
    print("create defect payload",payload)
    response = requests.post(url, json=payload, headers=headers, auth=auth)
    if response.status_code == 201:
        issue_key = response.json().get("key", "UNKNOWN")
        return {"message": f"New defect {issue_key} created successfully!"}
    else:
        raise HTTPException(status_code=response.status_code, detail=response.text)


def extract_project_key(user_query, email_id, access_token):
    """
    Extracts the project key from a user query using OpenAI.
    If unclear, fetches all projects and prompts the user.
    """
    response = openai_client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI assistant that extracts JIRA project keys from user queries."
                    " A JIRA project key is a short, uppercase alphanumeric identifier (e.g., 'SCRUM', 'DEVOPS')."
                    " Your task is to analyze the given user query and extract the correct JIRA project key if mentioned."
                    " If multiple project names appear, return only the most relevant one."
                    " If no project key is explicitly mentioned, return 'unknown' without additional text."
                )
            },
            {
                "role": "user",
                "content": (
                    "Analyze the following user query and extract the JIRA project key:\n\n"
                    f"'{user_query}'\n\n"
                    "### Instructions:\n"
                    "- Identify the project key based on standard JIRA naming conventions.\n"
                    "- A project key is typically a short, uppercase alphanumeric code (e.g., 'SCRUM', 'DEVOPS').\n"
                    "- If the query contains a project name (e.g., 'Scrum project'), infer its key by converting it to uppercase and ensuring it follows JIRA's convention.\n"
                    "- If multiple projects are mentioned, return the most relevant one based on context.\n"
                    "- If no valid project key is found, return only 'unknown' without any additional text or explanation."
                )
            }
        ],
        temperature=0.1
    )

    project_key = response.choices[0].message.content.strip()

    print("project_key",project_key)
    # If OpenAI can't determine the project, fetch all available projects
    if project_key.lower() == "unknown":
        all_projects = get_all_projects(email_id=email_id, access_token=access_token)
        print("all projects",all_projects)
        # ✅ Handle API Failure
        if all_projects is None:
            return None, {
                "message": "I couldn't retrieve your projects. Please check your JIRA credentials or permissions and try again."
            }

        # ✅ If the user has projects, prompt them to choose instead of erroring out
        if all_projects:
            project_list = "\n".join([f"🔹 {proj.get('key', 'Unknown')} → {proj.get('name', 'No Name')}" for proj in all_projects])
            return None, {
                "message": f"I couldn't identify a project from your query. Here are the projects in your JIRA account:\n\n{project_list}\n\n"
                           "Please specify which project you'd like to use."
            }

    return project_key, None


def analyze_response(user_query, api_response):
    """
    Uses OpenAI to analyze and summarize the API response based on the user's query.
    """
    response = openai_client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {"role": "system", "content": "You are an AI assistant that extracts key information from JIRA API responses and provides concise, user-friendly summaries."},
            {"role": "user", "content": f"""User query: {user_query}

            JIRA API Response:
            {json.dumps(api_response, indent=2)}

            Instructions:
            - Extract only the most relevant details based on the user query.
            - Summarize in clear, natural language.
            - If the response contains multiple issues, provide a structured list.
            - If the query is unclear, make an intelligent assumption.
            - Avoid unnecessary technical jargon unless relevant to the query.
            - If there’s no useful data, respond with: 'No relevant information found.' 
            """}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()

def determine_intent(user_query):
    """
    Uses OpenAI to classify the intent of the user query dynamically.
    Returns the detected intent.
    """
    response = openai_client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {"role": "system", "content": "You are an AI assistant that classifies user queries related to JIRA."},
            {"role": "user", "content": f"Classify this query: '{user_query}'. "
                                        "Return only one of the following intents: ['project_status', 'update_defect', 'create_defect', 'fetch_project_details', 'unknown']."}
        ],
        temperature=0
    )
    
    return response.choices[0].message.content.strip()

def extract_issue_details(user_query, previous_query, previous_response):
    """
    Extracts the project key, summary, description, and priority from a user query for JIRA issue creation.
    """
    response = openai_client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI assistant that extracts JIRA issue details from user queries. "
                    "Your task is to analyze the query and extract the following fields:\n"
                    "1. **Project Key**: A short uppercase identifier (e.g., 'SCRUM').\n"
                    "2. **Summary**: A brief title summarizing the issue.\n"
                    "3. **Description**: A detailed explanation of the issue.\n"
                    "4. **Status**: If the user specifies a status (e.g., 'To Do', 'In Progress', 'Done'), extract it; otherwise, return 'To Do'.\n"
                    "5. **Priority**: The priority level (e.g., 'High', 'Medium', 'Low').\n"
                    "6. **Defect ID** (only if the user wants to update an issue).\n\n"
                    "### Instructions:\n"
                    "- Extract and return the project key if mentioned; otherwise, return 'unknown'.\n"
                    "- The summary should be concise and relevant.\n"
                    "- The description should provide additional context if available.\n"
                    "- Infer priority if explicitly mentioned; otherwise, set it to 'Medium' by default.\n"
                    "- If the user intends to update a defect, extract the defect ID (a numeric identifier like 'BUG-1234')."
                ),
            },
            {
                "role": "user",
                "content": (
                    "consider the previous query and previous response: "
                    f"previous query : {previous_query}"
                    f"previous_response : {previous_response}"
                    f"Analyze the following user query and extract the required JIRA issue details:\n\n"
                    f"'{user_query}'\n\n"
                    "### Instructions:\n"
                    "- **Project Key:** Extract and return only the JIRA project key (uppercase format, e.g., 'SCRUM').\n"
                    "- **Summary:** Provide a short title summarizing the issue.\n"
                    "- **Description:** Extract any additional details beyond the summary.\n"
                    "- **Status**: If the user specifies a status (e.g., 'To Do', 'In Progress', 'Done'), extract it; otherwise, return 'To Do'.\n"
                    "- **Priority:** Identify priority as 'High', 'Medium', or 'Low' based on urgency keywords (e.g., 'urgent', 'ASAP', 'critical' → 'High').\n"
                    "- If any field is missing, return 'unknown'.\n"
                    "- Return a JSON object with fields: project_key, summary, description, priority."
                )
            }
        ],
        temperature=0.1
    )
    print("-----"*10)
    print(response.choices[0].message.content.strip())
    print("----"*10)
    ai_response = response.choices[0].message.content.strip()
    ai_response = re.sub(r"^```json|```$", "", ai_response).strip()
    extracted_details = json.loads(ai_response)
    return extracted_details


@app.post("/new_chat")
def process_query(request: UserQuery):
    user_query = request.query
    email_id = request.email_id
    access_token = request.access_token
    previous_query = request.previous_query
    previous_response = request.previous_response

    # Step 1: Dynamically determine intent
    intent = determine_intent(user_query)
    
    # Step 2: Extract the project key (if needed)
    # project_id, response_message = extract_project_key(user_query, email_id, access_token)
    # print("project id",project_id, response_message)
    # if response_message:  
    #     return response_message  # Ask user to pick a project instead of failing

    responses = []
    print("intent",intent)
    extracted_details = extract_issue_details(user_query, previous_query, previous_response)
    project_id = extracted_details.get("project_key", "unknown")
    summary = extracted_details.get("summary", "")
    description = extracted_details.get("description", "unknown")
    priority = extracted_details.get("priority", "unknown")
    status = extracted_details.get("status", "To Do")
    defect_id = extracted_details.get("defect_id")
    # Step 3: Handle the detected intent dynamically
    if intent == "project_status":
        raw_response = fetch_project_details(project_id, user_query, email_id, access_token)
        # analyzed_response = analyze_response(user_query, raw_response)
        # print("analyzed response",analyzed_response)
        responses.append({"message": raw_response})

    elif intent == "fetch_project_details":
        raw_response = fetch_project_details(project_id, user_query, email_id, access_token)
        # analyzed_response = analyze_response(user_query, raw_response)
        responses.append({"message": raw_response})

    elif intent == "update_defect":
        raw_response = update_defect_in_jira(
            defect_id=defect_id, 
            status=status,
            description=description,
            priority=priority,
            email_id=email_id,
            access_token=access_token
        )
        analyzed_response = analyze_response(user_query, raw_response)
        responses.append({"message": analyzed_response})

    elif intent == "create_defect":
        raw_response = create_defect_in_jira(
            project_key=project_id,
            summary=summary,
            description=description,
            priority=priority,
            email_id=email_id,
            access_token=access_token
        )
        analyzed_response = analyze_response(user_query, raw_response)
        responses.append({"message": analyzed_response})

    else:
        return {"message": "I couldn't determine what you need. Could you clarify your request?"}

    return responses

# Store state tokens to prevent CSRF attacks
state_store = {}

def generate_state_token():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=16))

@app.get("/jira/auth-url")
async def get_jira_auth_url():
    state = generate_state_token()
    state_store[state] = True  # Store the state token for validation
    auth_url = (
        f"{JIRA_AUTH_URL}?client_id={JIRA_CLIENT_ID}&response_type=code&redirect_uri={JIRA_REDIRECT_URI}"
        f"&scope={JIRA_API_SCOPE}&state={state}"
    )
    return {"auth_url": auth_url}

@app.api_route("/jira/callback", methods=["GET", "POST"])
async def jira_oauth_callback(request: Request):
    query_params = request.query_params  # Get query parameters from the request
    code = query_params.get("code")
    state = query_params.get("state")
    
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing OAuth2 parameters.")
    
    if state not in state_store:
        raise HTTPException(status_code=400, detail="Invalid state parameter.")
    
    # Exchange authorization code for access token
    token_payload = {
        "grant_type": "authorization_code",
        "client_id": JIRA_CLIENT_ID,
        "client_secret": JIRA_CLIENT_SECRET,
        "code": code,
        "redirect_uri": JIRA_REDIRECT_URI,
    }
    response = requests.post(JIRA_TOKEN_URL, json=token_payload, headers={"Content-Type": "application/json"})
    
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to get access token.")
    
    token_data = response.json()
    access_token = token_data.get("access_token")
    
    if not access_token:
        print("Token response:", token_data)  # Log the token response for debugging
        raise HTTPException(status_code=400, detail="Invalid token response.")
    
    # Retrieve the cloud ID
    cloud_id_response = requests.get(
        "https://api.atlassian.com/oauth/token/accessible-resources",
        headers={"Authorization": f"Bearer {access_token}"}
    )
    
    if cloud_id_response.status_code != 200:
        print("Cloud ID response:", cloud_id_response.text)  # Log the cloud ID response for debugging
        raise HTTPException(status_code=cloud_id_response.status_code, detail="Failed to get cloud ID.")
    
    cloud_id_data = cloud_id_response.json()
    if not cloud_id_data or not isinstance(cloud_id_data, list) or not cloud_id_data[0].get("id"):
        print("Cloud ID response:", cloud_id_data)  # Log the cloud ID response for debugging
        raise HTTPException(status_code=400, detail="Invalid cloud ID response.")
    
    cloud_id = cloud_id_data[0]["id"]
    
    # Retrieve the authenticated user's details
    user_response = requests.get(
        "https://api.atlassian.com/me",
        headers={"Authorization": f"Bearer {access_token}"}
    )
    
    if user_response.status_code != 200:
        print("User response:", user_response.text)  # Log the user response for debugging
        raise HTTPException(status_code=user_response.status_code, detail="Failed to get user details.")
    
    user_data = user_response.json()
    user_name = user_data.get("name")
    
    # Redirect to the frontend with the access token, cloud ID, and user name
    frontend_redirect_url = f"http://localhost:5173?access_token={access_token}&cloud_id={cloud_id}&user_name={user_name}"
    return RedirectResponse(url=frontend_redirect_url)


if __name__ == "__main__":
    uvicorn.run(app, port=7000)
