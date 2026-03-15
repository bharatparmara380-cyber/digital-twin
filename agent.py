# ─────────────────────────────────────────────────────────────────────────────
#  agent.py  –  The core Digital Twin agent
#
#  WHAT IS A LANGCHAIN AGENT?
#  A regular LLM call is stateless: you send a prompt, get a response, done.
#  An AGENT is an LLM that can:
#    1. Reason about what steps to take
#    2. Decide which tool to call
#    3. Observe the tool's output
#    4. Repeat until it has enough info to answer
#
#  This is called the ReAct loop (Reasoning + Acting):
#
#    User Question
#        ↓
#    [Think] "What do I need to answer this?"
#        ↓
#    [Act]   Call a tool (e.g., resume_search)
#        ↓
#    [Observe] Read the tool's output
#        ↓
#    [Think] "Do I have enough? If yes → answer. If no → call another tool."
#        ↓
#    Final Answer
#
#  This loop can repeat multiple times in one user message!
# ─────────────────────────────────────────────────────────────────────────────

from langchain_groq import ChatGroq
# ChatGroq is a LangChain wrapper around Groq's API.
# It gives us a standard LLM interface so we can swap providers easily.

from langchain.agents import AgentExecutor, create_react_agent
# create_react_agent(): builds the ReAct reasoning loop
# AgentExecutor():     runs the loop and handles tool calls

from langchain.memory import ConversationBufferWindowMemory
# Stores the last N conversation turns so the agent remembers context.
# "Window" means it forgets old turns to avoid hitting token limits.

from langchain import hub
# LangChain Hub hosts reusable prompts. We'll pull the standard ReAct prompt.

from langchain.prompts import PromptTemplate

import config


# ─── System Prompt ────────────────────────────────────────────────────────────

def build_system_prompt() -> str:
    """
    The system prompt defines the agent's identity and behavior.
    This is the most important part of making the agent feel like YOU.
    Customize the personality section freely!
    """
    return f"""You are the digital twin of {config.YOUR_NAME} — {config.YOUR_TAGLINE}.

You speak in first person ("I", "my", "me") as if you ARE {config.YOUR_NAME}.
You are warm, professional, knowledgeable, and genuinely helpful.

## Your Capabilities
You have access to four tools:
1. resume_search    → Search your personal resume/bio for background info
2. calculator       → Perform any math calculation
3. web_search       → Look up current information from the internet
4. wikipedia_search → Look up facts, concepts, or definitions

## How to Behave
- When asked about yourself (skills, experience, education, projects):
  ALWAYS use the resume_search tool first to give accurate, grounded answers.
- When asked to calculate something: use the calculator tool.
- When asked about current events, news, or facts you are unsure about:
  use web_search or wikipedia_search.
- Be concise but thorough. Answer in a natural, human tone.
- If you don't know something and can't find it with tools, say so honestly.
- Never make up facts about yourself — only state what's in the resume.

## Personality
- Professional but approachable
- Enthusiastic about technology and problem solving
- Happy to help with coding, career advice, or general questions
- Always refer to yourself as {config.YOUR_NAME}

Remember: You ARE {config.YOUR_NAME}. Talk like a real person, not a robot.
"""


# ─── ReAct Prompt Template ───────────────────────────────────────────────────

# The ReAct framework requires a very specific prompt format.
# The LLM must produce output in this pattern:
#
#   Thought: I need to look up the user's work experience
#   Action: resume_search
#   Action Input: work experience
#   Observation: [tool output]
#   Thought: I now have enough info
#   Final Answer: I worked at ...

REACT_PROMPT_TEMPLATE = """{system_prompt}

You have access to the following tools:
{tools}

Use the following format EXACTLY:

Question: the input question you must answer
Thought: think about what to do
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Previous conversation history:
{chat_history}

Question: {input}
Thought: {agent_scratchpad}"""


# ─── Build the Agent ─────────────────────────────────────────────────────────

def build_agent(tools: list) -> AgentExecutor:
    """
    Assembles the complete Digital Twin agent.

    Args:
        tools: List of tool objects from tools.get_all_tools()

    Returns:
        An AgentExecutor ready to receive user messages
    """
    config.validate_config()  # fail fast if API key is missing

    # ── 1. Initialize the LLM ──────────────────────────────────────────────
    print(f"🤖  Initializing Groq LLM: {config.GROQ_MODEL}")

    llm = ChatGroq(
        api_key=config.GROQ_API_KEY,
        model=config.GROQ_MODEL,
        temperature=0.5,
        # temperature controls randomness:
        #   0.0 = deterministic, robotic
        #   0.7 = natural, slightly creative (good for conversation)
        #   1.0 = very random/creative
        max_tokens=2048,
    )

    # ── 2. Build the Prompt ────────────────────────────────────────────────
    prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)

    # Inject the system prompt (our custom identity) into the template
    prompt = prompt.partial(system_prompt=build_system_prompt())

    # ── 3. Create the ReAct Agent ──────────────────────────────────────────
    # create_react_agent() wires together:
    #   - llm: the language model that reasons
    #   - tools: the actions it can take
    #   - prompt: the instructions on how to reason
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    # ── 4. Set Up Conversation Memory ──────────────────────────────────────
    # ConversationBufferWindowMemory keeps the last `k` turns.
    # k=10 means it remembers the last 10 exchanges (5 from each side).
    # memory_key must match the {chat_history} placeholder in the prompt.
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=False,  # return as a formatted string, not Message objects
        k=10,
    )

    # ── 5. Wrap in AgentExecutor ────────────────────────────────────────────
    # AgentExecutor is the "runner" that:
    #   - Feeds the user message to the agent
    #   - Parses "Action: calculator" → calls the actual calculator function
    #   - Feeds the tool output back as "Observation: ..."
    #   - Stops when the agent says "Final Answer: ..."
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        # verbose=True prints every Thought/Action/Observation step.
        # This is great for learning and debugging!
        # Set to False in production to hide internal reasoning.
        handle_parsing_errors=True,
        # If the LLM output is malformed, retry instead of crashing.
        max_iterations=10,
        # Safety limit: stop after 6 tool calls to prevent infinite loops.
    )

    print("✅  Digital Twin agent is ready!\n")
    return agent_executor


# ─── Chat Helper ─────────────────────────────────────────────────────────────

def chat(agent_executor: AgentExecutor, user_message: str) -> str:
    """
    Send a message to the agent and get a response.

    Args:
        agent_executor: The built agent from build_agent()
        user_message:   The user's text input

    Returns:
        The agent's response as a string
    """
    try:
        # agent_executor.invoke() runs the full ReAct loop.
        # It returns a dict with "output" as the final answer.
        response = agent_executor.invoke({"input": user_message})
        return response["output"]
    except Exception as e:
        return f"Sorry, I ran into an error: {e}. Please try again."
