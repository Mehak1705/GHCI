from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import json
import re
# --- Step 1: Set up Gemini API key ---
os.environ["GOOGLE_API_KEY"] = "Enterkey"

# --- Step 2: Initialize Gemini model ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# --- Step 3: Define the prompt template ---
prompt = ChatPromptTemplate.from_template("""
You are FairSync Explainable AI Assistant — a regulatory-aware conversational model designed for transparent banking decisions.

Return your results in a structured format.
decision_explaination is a 50 words response of what happened and why it happened. Keep the language simple which can be understood by a non financial person.
key factors are for metrics which incluenced the decision
what_if_guidance is a 50 words condensed actionable guidance. Do not use brakets, tone - conversational. If this is about contacting us then skip to support.
{{
    "decision_summary_explanation": "",
    "key_factors": [],
    "what_if_guidance": "",
    "support": "For more support you can call our toll free number +1 (863) 281-4984 or request for us to call you back."
}}

User Input:
{input_text}

Generate meaningful explanation based on detected loan attributes.
""")

# --- Step 4: Build the chain ---
chain = prompt | llm | StrOutputParser()

# --- Step 5: Run the agent ---
def extract_information(user_query: str):
    result = chain.invoke({"input_text": user_query})
    return result


def extract_fields(raw_response: str):
    """
    Extract decision_summary_explanation, what_if_guidance, and support
    from raw LLM response, even if it's not valid JSON.
    """
    fields = {
        "decision_summary_explanation": "",
        "what_if_guidance": "",
        "support": ""
    }

    # Regex patterns for each field
    patterns = {
        "decision_summary_explanation": r'"decision_summary_explanation"\s*:\s*"(.*?)"',
        "what_if_guidance": r'"what_if_guidance"\s*:\s*"(.*?)"',
        "support": r'"support"\s*:\s*"(.*?)"'
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, raw_response, re.DOTALL)
        if match:
            fields[key] = match.group(1).strip()

    return fields

if __name__ == "__main__":
    user_input = input("Enter your query: ")
    info = extract_information(user_input)
    try:
        info_json = json.loads(info)
    except json.JSONDecodeError:
        info_json = extract_fields(info)
    output = (
    f"{info_json.get('decision_summary_explanation', '')} "
    f"{info_json.get('what_if_guidance', '')} "
    f"{info_json.get('support', '')}")
    print("\nResponse:\n", info)
    print("\nResponse:\n", output)
