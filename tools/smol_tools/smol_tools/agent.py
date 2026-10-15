# [MODULE DOC] Tool-calling agent for the smol_tools desktop demo: a few @tool-decorated functions (random
# number, time,
# web browser, weather) exposed to a transformers CodeAgent driven by a local SmolLM2-1.7B GGUF via
# llama-cpp. Network use (wttr.in) and opening the system browser are side effects of the tools.
from .base import SmolTool
from typing import Generator, List, Dict, Any, Callable
import json
import re
from datetime import datetime
import random
from transformers import tool, CodeAgent
import requests
import webbrowser

@tool
def get_random_number_between(min: int, max: int) -> str:
    # [FUNCTION DOC] Agent tool: returns a uniformly random integer in [min, max] (inclusive) formatted as a
    # string.
    """
    Gets a random number between min and max.

    Args:
        min: The minimum number.
        max: The maximum number.

    Returns:
        A random number between min and max.
    """
    return f"{random.randint(min, max):d}"


@tool
def get_weather(city: str) -> str:
    # [FUNCTION DOC] Agent tool: GETs "https://wttr.in/<city>?format=+%C,+%t" and returns a sentence with
    # the condition (text
    # before the comma) and the temperature (text after the comma minus its last 2 chars, i.e. the unit).
    # KNOWN ISSUE: no timeout/error handling or URL-encoding of city; a response without a comma raises
    # IndexError; it reports the current temperature as "a high of".
    """
    Returns the weather forecast for a given city.

    Args:
        city: The name of the city.

    Returns:
        A string with a mock weather forecast.
    """
    url = 'https://wttr.in/{}?format=+%C,+%t'.format(city)
    res = requests.get(url).text

    return f"The weather in {city} is {res.split(',')[0]} with a high of {res.split(',')[1][:-2]} degrees Celsius."

@tool
def get_current_time() -> str:
    # [FUNCTION DOC] Agent tool: returns "The current time is H:M." using local time.
    # KNOWN ISSUE: minutes/hours are not zero-padded (docstring promises HH:MM) and datetime.now() is called
    # twice, so the hour and minute can come from different instants.
    """
    This is a tool that returns the current time.
    It returns the current time as HH:MM.
    """
    return f"The current time is {datetime.now().hour}:{datetime.now().minute}."

@tool
def open_webbrowser(url: str) -> str:
    # [FUNCTION DOC] Agent tool: opens url in the default web browser (side effect) and returns a
    # confirmation sentence with
    # "https://" and "www." removed from the url.
    """
    This is a tool that opens a web browser to the given website.
    If the user asks to open a website or a browser, you should use this tool.

    Args:
        url: The url to open.
    """
    webbrowser.open(url)
    return f"I opened {url.replace('https://', '').replace('www.', '')} in the browser."


class SmolToolAgent(SmolTool):
    # [CLASS DOC] SmolTool that answers a request by letting a CodeAgent pick and call the module's tools.
    # Uses the shared
    # cached SmolLM2-1.7B-Instruct GGUF model.
    def __init__(self):
        # [FUNCTION DOC] Registers the four tools, builds a name -> tool mapping (toolbox) and a CodeAgent
        # using llm_engine and the
        # system prompt, then initializes the base class (loads/caches the model and warms up on first load).
        self.tools = [get_random_number_between, get_current_time, open_webbrowser, get_weather]
        self.toolbox = {tool.name: tool for tool in self.tools}
        self.json_code_agent = CodeAgent(tools=self.tools, llm_engine=self.llm_engine, system_prompt=self._get_system_prompt())
        super().__init__(
            model_repo="andito/SmolLM2-1.7B-Instruct-F16-GGUF",
            model_filename="smollm2-1.7b-8k-dpo-f16.gguf",
            system_prompt=self._get_system_prompt(),
            prefix_text=""
        )

    def llm_engine(self, messages, stop_sequences=["Task", "<|endoftext|>"]) -> str:
        # [FUNCTION DOC] LLM callback for the CodeAgent: streams a greedy chat completion (max 2048 tokens,
        # temperature 0) from the
        # llama-cpp model for the given messages and returns the concatenated text, stopping at
        # "<end_action>" or
        # "<|endoftext|>" chunks.
        # KNOWN ISSUE: stop_sequences is accepted (with a mutable list default) but ignored.
        output = ""
        for chunk in self.model.create_chat_completion(
            messages=messages,
            max_tokens=2048,
            temperature=0.0,
            top_p=1.0,
            top_k=50,
            repeat_penalty=1.0,
            stream=True
        ):
            content = chunk['choices'][0]['delta'].get('content')
            if content:
                if content in ["<end_action>", "<|endoftext|>"]:
                    break
                output += content
        return output

    def _get_system_prompt(self) -> str:
        # [FUNCTION DOC] HELPER: returns the function-calling system prompt template (with CodeAgent
        # placeholders for tool
        # descriptions, managed agents and authorized imports) asking for a JSON list inside <tool_call> tags.
        return """You are an expert in composing functions. You are given a question and a set of possible functions.
Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out and refuse to answer.
If the given question lacks the parameters required by the function, also point it out.

You have access to the following tools:
<<tool_descriptions>>

<<managed_agents_descriptions>>

You can use imports in your code, but only from the following list of modules: <<authorized_imports>>

The output MUST strictly adhere to the following format, and NO other text MUST be included.
The example format is as follows. Please make sure the parameter type is correct. If no function call is needed, please make the tool calls an empty list '[]'.
<tool_call>[
{"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}},
... (more tool calls as required)
]</tool_call>"""

    def _parse_response(self, text: str) -> List[Dict[str, Any]]:
        # [FUNCTION DOC] HELPER: finds the first "<tool_call>...</tool_call>" block (dot matches newline)
        # and returns its
        # JSON-decoded content (a list of {"name", "arguments"} mappings); returns the original text if none
        # found.
        # Raises json.JSONDecodeError on invalid JSON. (Return type annotation says list, but may return the
        # str.)
        pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return json.loads(matches[0])
        return text

    def _call_tools(self, tool_calls: List[Dict[str, Any]]) -> List[str]:
        # [FUNCTION DOC] HELPER: for each tool call, invokes the matching toolbox tool with its "arguments"
        # as keyword args and
        # collects the result, or "Tool <name> not found." for unknown names. Returns the list of results.
        tool_responses = []
        for tool_call in tool_calls:
            if tool_call["name"] in self.toolbox:
                tool_responses.append(
                    self.toolbox[tool_call["name"]](**tool_call["arguments"])
                )
            else:
                tool_responses.append(f"Tool {tool_call['name']} not found.")
        return tool_responses

    def process(self, text: str) -> Generator[str, None, None]:
        # [FUNCTION DOC] Runs the CodeAgent on text (returning generated code), parses tool calls from it;
        # if there are none (or
        # parsing/calling raises, which is printed) yields the raw agent response once; otherwise executes
        # the calls
        # and yields each tool result as a string.
        response = self.json_code_agent.run(text, return_generated_code=True)
        # Parse and execute the tool calls
        try:
            tool_calls = self._parse_response(response)
            if tool_calls in [response, [], ""]:
                yield response
                return
            tool_responses = self._call_tools(tool_calls)
        except Exception as e:
            print("error", e)
            yield response
            return

        # Yield each tool response
        for response in tool_responses:
            yield str(response)