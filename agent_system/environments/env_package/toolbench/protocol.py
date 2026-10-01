"""Shared single-action protocol for preprocessing and live ToolBench prompts."""

ACTION_RULES = """Interaction rules for this run:
Reply with exactly one action, using these two lines:
Action: <one exact available function name>
Action Input: <one JSON object>
Stop immediately after the JSON object. Do not write observations, tool responses,
another action, Markdown fences, or a simulated conversation. The environment
will execute the action and return the real result in the next user message.
To submit the completed answer use:
Action: Finish
Action Input: {"return_type": "give_answer", "final_answer": "<your complete answer>"}
To give up use:
Action: Finish
Action Input: {"return_type": "give_up_and_restart"}
"""

# String stops require detokenize=True in vLLM.
ACTION_STOPS = ["\nObservation:", "\nTool response", "\nPrevious action:", "\nUser query:"]


def initial_observation(system_prompt, query):
    return f"{str(system_prompt or '').strip()}\n\n{ACTION_RULES}\nUser query:\n{str(query or '').strip()}\nBegin!"
