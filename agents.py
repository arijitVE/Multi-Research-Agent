from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search , web_scrape 
from dotenv import load_dotenv
import os

load_dotenv()

#model setup 
# llm = ChatOpenAI(model = "gpt-4o-mini",temperature=0)

def get_openai_api_key():
    try:
        import streamlit as st
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in Streamlit secrets or environment variables")

    return api_key

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=get_openai_api_key())


classifier_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a query router. Decide if a query needs deep research or can be answered directly.

Reply with ONLY one of these two words:
- SIMPLE  → for greetings, definitions, quick facts, math, basic how-to questions
- RESEARCH → for topics needing current data, analysis, comparisons, trends, or detailed reports

No explanation. One word only."""),
    ("human", "Query: {query}")
])

classifier_chain = classifier_prompt | llm | StrOutputParser()

#1st agent 
def build_search_agent():
    return create_agent(
        model = llm,
        tools= [web_search]
    )

#2nd agent 

def build_reader_agent():
    return create_agent(
        model = llm,
        tools = [web_scrape]
    )


#writer chain 

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

#critic_chain 

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()
