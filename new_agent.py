from langgraph.graph import MessageGraph, StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from typing import Annotated, TypedDict, List, Union, Optional
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate
import os
import base64
import pandas as pd

# LLM setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="deepseek-r1-distill-llama-70b", api_key=GROQ_API_KEY)

class AgentState(TypedDict):
    messages: Annotated[List[str], add_messages]
    image_path : str 
    memory: dict

# Define the function to check the type of query
def classify_query(state: AgentState):
    query = state["messages"][-1]
    prompt = """
    Classify the following query into one of the following categories:
    - 'broken_lcd' if it is related to a broken LCD screen.
    - 'lcd_review' if the user is asking to review an image where a tool detected LCD damage but the customer disagrees.
    - 'motherboard_issue' if it is related to a system not powering on or a motherboard failure.
    - 'general_support' for other issues.
    
    Query: {query}
    Reply with only one of the category names.
    """
    formated_prompt = prompt.format(query=query)
    response = llm.invoke([HumanMessage(content=formated_prompt)])
    state["classify"] = response.content
    print("----------",response.content)
    return state

# Create the LangGraph workflow
workflow = StateGraph(AgentState)

# Add the classify_query function to the workflow
workflow.add_node("classify_query", classify_query)

# Define the entry and exit points of the workflow
workflow.add_edge(START, "classify_query")
workflow.add_edge("classify_query", END)

# Compile the workflow
graph = workflow.compile()

# Example test case: Broken LCD query without image
state = {
    "messages": [
        {
            "role": "user", 
         "content": "Hello, the customer provided me this picture of a broken LCD and I need to know what to do."
        }
    ]
}

response = graph.invoke(state)

def chatbot():
    # Initialize state with empty messages, no image, and empty memory
    state: AgentState = {
        "messages": [],
        "image_path": "",
        "memory": {},
        "classify": ""
    }
    print("Chatbot started. Type 'exit' to quit.")
    
    while True:
        # Get user input
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break
        
        # Check if we are expecting an image input:
        # If the assistant has already asked for an image and no image has been provided.
        if state.get("classify") == "broken_lcd" and not state["image_path"]:
            # Assume the user now provides an image path or a base64-encoded string.
            state["image_path"] = user_input.strip()
            # Log the image input as a message for record
            state["messages"].append({"role": "user", "content": f"[Image provided: {state['image_path']}]"})
            print("Assistant: Thank you. I will now analyze the image.")
            # Here, you would typically call your image analysis function.
            # For this example, we simply reset the classification.
            state["classify"] = ""
            continue

        # Append the new user message
        state["messages"].append({"role": "user", "content": user_input})
        
        # Run the workflow from the current state
        state = graph.invoke(state)
        # Print the latest assistant message
        if state["messages"]:
            last_message = state["messages"][-1]["content"]
            print("Assistant:", last_message)

# # Run the chatbot
# if __name__ == "__main__":
#     chatbot()

# Define planner agent
def planner(state: AgentState):
    query = state["messages"][-1]
    prompt = """
    Plan the next steps based on the following query:{query}
    Based on the query you should call the respective team to resolve the issue.

    If the query is related to a broken LCD screen, recommend the user to send the laptop for repair.
    If the query is related to a system not powering on or a motherboard failure, recommend the user to check the power supply and contact support.
    If the query is related to a review of an image where a tool detected LCD damage but the customer disagrees, recommend the user to send the laptop for repair.
    If the query is related to general support, recommend the user to contact support for further assistance.

    """
    formated_prompt = prompt.format(query=query)
    response = llm.invoke([HumanMessage(content=formated_prompt)])
    return response.content

import openai
client = openai.OpenAI(api_key=GROQ_API_KEY)
# Define the function to process the image
def image_process(state: AgentState):
    # Read and encode the image
    with open(state["image_path"], "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
   
    # LLM prompt for damage analysis
    prompt = ChatPromptTemplate.from_template(
        "Analyze the following image and user query to determine if the laptop screen is broken:\n"
        "User Query: '{query}'\n"
        "Image: [provided as base64]\n"
        "Provide a brief explanation and conclude with 'Yes, it's broken' or 'No, it's not broken'."
    )
    chain = prompt | llm
   
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in analyzing laptop screen damage."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt.format(query=state["query"])},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]
            }
        ],
        max_tokens=200
    )
    state["damage_analysis"] = response.choices[0].message.content
    return state

# Analyse the reason for screen damage and check in the document its covered under warranty or not.
def screen_damage_analysis(state: AgentState):
    query = state["messages"][-1]
    prompt = """
    Analyse the reason for screen damage and check in the document its covered under warranty or not.

    """
    formated_prompt = prompt.format(query=query)
    response = llm.invoke([HumanMessage(content=formated_prompt)])
    return response.content

# Agent 3: SFDC Checker Agent
def check_sfdc(state: AgentState) -> AgentState:
    # Path to the SFDC Excel file (SERVICE_TAG, SKU)
    sfdc_file_path = r"D:\AgenticAI\LCD prototype\SFDC.xlsx"  # Replace with actual path to SFDC file
    sfdc_data = pd.read_excel(sfdc_file_path)  # Load SFDC Excel file
    service_tag = state["service_tag"]
    
    sku_row = sfdc_data[sfdc_data["SERVICE_TAG"] == service_tag]
    sku_status = "Yes" if not sku_row.empty and sku_row["SKU"].iloc[0] == "Yes" else "No"
    # print (sku_status)
    prompt = ChatPromptTemplate.from_template(
        "Given the service tag '{service_tag}', the SKU status in the SFDC system is '{sku_status}'. "
        "Provide a natural language response summarizing this finding."
    )
    chain = prompt | llm
    response = chain.invoke({"service_tag": service_tag, "sku_status": sku_status})
    state["sku_status"] = response.content
    return state

# Agent 4: Warranty Claim Checker Agent
def check_warranty_claim(state: AgentState) -> AgentState:
    # Path to the Warranty Excel file (SERVICE_TAG, WARRANTY_TAKEN)
    warranty_file_path = r"D:\AgenticAI\LCD prototype\Warranty.xlsx"  # Replace with actual path to Warranty file
    warranty_data = pd.read_excel(warranty_file_path)  # Load Warranty Excel file
    service_tag = state["service_tag"]
    
    claim_row = warranty_data[warranty_data["SERVICE_TAG"] == service_tag]
    claim_status = 1 if not claim_row.empty and claim_row["WARRANTY_TAKEN"].iloc[0] == 1 else 0
    
    prompt = ChatPromptTemplate.from_template(
        "Given the service tag '{service_tag}' and SKU status '{sku_status}', "
        "the warranty claim status is {claim_status} (1 means already claimed, 0 means not claimed). "
        "If the SKU status includes 'Yes', determine if the warranty can be claimed again. "
        "Provide a natural language response summarizing this."
    )
    chain = prompt | llm
    response = chain.invoke({
        "service_tag": service_tag,
        "sku_status": state["sku_status"],
        "claim_status": claim_status
    })
    state["warranty_claim_status"] = response.content
    return state
