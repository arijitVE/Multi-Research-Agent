# from langchain.agents import create_agent
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from tools import web_search , web_scrape 
# from dotenv import load_dotenv
# import os

# load_dotenv()

# #model setup 
# # llm = ChatOpenAI(model = "gpt-4o-mini",temperature=0)

# def get_openai_api_key():
#     try:
#         import streamlit as st
#         return st.secrets["OPENAI_API_KEY"]
#     except Exception:
#         pass

#     api_key = os.getenv("OPENAI_API_KEY")

#     if not api_key:
#         raise ValueError("OPENAI_API_KEY not found in Streamlit secrets or environment variables")

#     return api_key

# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=get_openai_api_key())


# classifier_prompt = ChatPromptTemplate.from_messages([
#     ("system", """You are a query router. Decide if a query needs deep research or can be answered directly.

# Reply with ONLY one of these two words:
# - SIMPLE  → for greetings, definitions, quick facts, math, basic how-to questions
# - RESEARCH → for topics needing current data, analysis, comparisons, trends, or detailed reports

# No explanation. One word only."""),
#     ("human", "Query: {query}")
# ])

# classifier_chain = classifier_prompt | llm | StrOutputParser()

# #1st agent 
# def build_search_agent():
#     return create_agent(
#         model = llm,
#         tools= [web_search]
#     )

# #2nd agent 

# def build_reader_agent():
#     return create_agent(
#         model = llm,
#         tools = [web_scrape]
#     )


# #writer chain 

# writer_prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
#     ("human", """Write a detailed research report on the topic below.

# Topic: {topic}

# Research Gathered:
# {research}

# Structure the report as:
# - Introduction
# - Key Findings (minimum 3 well-explained points)
# - Conclusion
# - Sources (list all URLs found in the research)

# Be detailed, factual and professional."""),
# ])

# writer_chain = writer_prompt | llm | StrOutputParser()

# #critic_chain 

# critic_prompt = ChatPromptTemplate.from_messages([
#      ("system", "You are a sharp and constructive research critic. Be honest and specific."),
#     ("human", """Review the research report below and evaluate it strictly.

# Report:
# {report}

# Respond in this exact format:

# Score: X/10

# Strengths:
# - ...
# - ...

# Areas to Improve:
# - ...
# - ...

# One line verdict:
# ..."""),
# ])

# critic_chain = critic_prompt | llm | StrOutputParser()



from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import (
    arxiv_search,
    financial_data,
    news_search,
    weather_search,
    web_scrape,
    web_search,
    wikipedia_search,
)
from dotenv import load_dotenv

load_dotenv()

#model setup 
llm = ChatOpenAI(model = "gpt-4o-mini",temperature=0)


classifier_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a query router. Decide if a query needs deep research or can be answered directly.

Reply with ONLY one of these five words:
- SIMPLE    → greetings, definitions, math, basic how-to
- FINANCE   → stock prices, crypto, gold, forex, market data
- NEWS      → current events, breaking news, today's updates
- ACADEMIC  → research papers, scientific topics, ML, physics
- RESEARCH  → everything else needing deep multi-source analysis

No explanation. One word only."""),
    ("human", "Query: {query}")
])

classifier_chain = classifier_prompt | llm | StrOutputParser()

#1st agent 
def build_search_agent():
    return create_agent(
        model = llm,
        tools= [web_search, news_search, wikipedia_search, arxiv_search, financial_data, weather_search]
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


# ── Chat chain (memory & chat history) ───────────────────────────────────────
# Takes the full report as system context + the rolling conversation history.
# Called turn-by-turn from app.py — history is managed in st.session_state.

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert research assistant. You have just produced the research report below.
Answer the user's questions based on this report and your broader knowledge.
Be specific, cite details from the report where relevant, and stay focused on the topic.

--- RESEARCH REPORT ---
{report}
--- END OF REPORT ---"""),
    ("placeholder", "{history}"),
    ("human", "{question}"),
])

chat_chain = chat_prompt | llm | StrOutputParser()


query_decomposer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a research query analyzer. Given a research topic,
generate 3 distinct search angles that together would cover the topic comprehensively.

Respond ONLY with a JSON object in this exact format, no explanation:
{{
  "primary": "main search query",
  "secondary": "different angle or subtopic query",
  "tertiary": "recent news or latest developments query"
}}"""),
    ("human", "Topic: {topic}")
])

query_decomposer_chain = query_decomposer_prompt | llm | StrOutputParser()


evaluator_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a strict research data quality evaluator.
Evaluate whether the gathered research data is sufficient to write
a high-quality, accurate report.

Score the data on these criteria:
- Relevance: Is the data related to the topic at all? (0-25 points)
- Coverage: Does it cover at least 2 angles? (0-25 points)
- Recency: Is there any recent information? (0-25 points)
- Depth: Is there more than surface-level snippets? (0-25 points)

Respond ONLY with a JSON object, no explanation:
{{
  "score": <total 0-100>,
  "relevance": <0-25>,
  "coverage": <0-25>,
  "recency": <0-25>,
  "depth": <0-25>,
  "passed": <true if score >= 40, false otherwise>,
  "missing": "<one sentence: what key information is missing>",
  "retry_query": "<if not passed: a better search query to fill the gap, else empty string>"
}}"""),
    ("human", """Topic: {topic}

Gathered data:
{data}""")
])

evaluator_chain = evaluator_prompt | llm | StrOutputParser()


improver_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert research writer tasked with improving
an existing research report based on specific instructions.
Maintain the same structure (Introduction, Key Findings, Conclusion, Sources)
but improve the content according to the instructions.
Be detailed, factual and professional."""),
    ("human", """Original report:
{report}

Critic feedback:
{feedback}

Improvement instructions:
{instructions}

Write the improved version of the complete report:""")
])

improver_chain = improver_prompt | llm | StrOutputParser()
