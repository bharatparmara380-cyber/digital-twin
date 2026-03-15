import math
import re
import requests
import wikipedia
from duckduckgo_search import DDGS
from langchain.tools import tool
import config


@tool
def calculator(expression: str) -> str:
    """
    Use this tool to perform mathematical calculations.
    Input should be a valid math expression like:
    '2 + 2', '15 * 0.18', 'sqrt(144)', '2 ** 10'
    """
    safe_names = {
        "sqrt": math.sqrt, "log": math.log,
        "sin": math.sin, "cos": math.cos,
        "tan": math.tan, "pi": math.pi,
        "e": math.e, "abs": abs, "round": round,
        "pow": pow, "floor": math.floor, "ceil": math.ceil,
    }
    cleaned = re.sub(r"[^0-9+\-*/().%, a-z_]", "", expression.lower())
    try:
        result = eval(cleaned, {"__builtins__": {}}, safe_names)
        return str(round(float(result), 6))
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool
def web_search(query: str) -> str:
    """
    Use this tool to search the internet for current information,
    recent news, or anything not in the knowledge base.
    Input should be a clear search query string.
    """
    try:
        from googlesearch import search
        import requests
        from bs4 import BeautifulSoup

        results = list(search(query, num_results=4, advanced=True))

        if not results:
            return "No results found. Try rephrasing."

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"Result {i}: {r.title}\n"
                f"Source: {r.url}\n"
                f"Summary: {r.description}\n"
            )
        return "\n".join(formatted)

    except Exception as e:
        return f"Web search failed: {e}"

@tool
def wikipedia_search(topic: str) -> str:
    """
    Use this tool to look up factual information, definitions,
    historical facts, or explanations of concepts from Wikipedia.
    Input should be the name of a person, place, or concept.
    """
    try:
        summary = wikipedia.summary(topic, sentences=5, auto_suggest=True)
        return f"Wikipedia – {topic}:\n{summary}"
    except wikipedia.exceptions.DisambiguationError as e:
        options = e.options[:5]
        return (
            f"'{topic}' is ambiguous. Did you mean:\n"
            + "\n".join(f"  - {opt}" for opt in options)
        )
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for '{topic}'."
    except Exception as e:
        return f"Wikipedia lookup failed: {e}"


@tool
def get_weather(city: str) -> str:
    """
    Use this tool when the user asks about weather, temperature,
    or forecast in any city or location.
    Input should be the city name like 'Mandi Himachal Pradesh'
    or 'Mumbai' or 'Delhi'.
    Returns actual current temperature and weather conditions.
    """
    try:
        city_encoded = city.replace(" ", "+")
        url = f"https://wttr.in/{city_encoded}?format=j1"
        response = requests.get(url, timeout=10)
        data = response.json()

        current   = data["current_condition"][0]
        area      = data["nearest_area"][0]
        temp_c    = current["temp_C"]
        temp_f    = current["temp_F"]
        feels     = current["FeelsLikeC"]
        humidity  = current["humidity"]
        desc      = current["weatherDesc"][0]["value"]
        wind      = current["windspeedKmph"]
        city_name = area["areaName"][0]["value"]
        region    = area["region"][0]["value"]

        return (
            f"Weather in {city_name}, {region}:\n"
            f"  Condition  : {desc}\n"
            f"  Temperature: {temp_c}C / {temp_f}F\n"
            f"  Feels Like : {feels}C\n"
            f"  Humidity   : {humidity}%\n"
            f"  Wind Speed : {wind} km/h\n"
        )
    except requests.exceptions.ConnectionError:
        return "No internet connection. Cannot fetch weather."
    except Exception as e:
        return f"Weather fetch failed: {e}"


def create_resume_search_tool(retriever):
    """Creates the resume RAG search tool tied to the retriever."""

    @tool
    def resume_search(query: str) -> str:
        """
        Use this tool to answer ANY question about the person's
        background, education, marks, skills, work experience,
        projects, certifications, or anything from their resume.
        ALWAYS use this first for personal questions.
        Input: a question about the person.
        """
        try:
            docs = retriever.invoke(query)
            if not docs:
                return "No relevant information found in the resume."
            context_parts = []
            for i, doc in enumerate(docs, 1):
                page = doc.metadata.get("page", "?")
                context_parts.append(
                    f"[Excerpt {i} — Page {page}]\n{doc.page_content.strip()}"
                )
            return "\n\n".join(context_parts)
        except Exception as e:
            return f"Resume search failed: {e}"

    return resume_search


def get_all_tools(retriever) -> list:
    """Returns all tools for the agent."""
    resume_tool = create_resume_search_tool(retriever)

    tools = [
        resume_tool,
        calculator,
        web_search,
        wikipedia_search,
        get_weather,
    ]

    print(f"Loaded {len(tools)} tools: {[t.name for t in tools]}")
    return tools
