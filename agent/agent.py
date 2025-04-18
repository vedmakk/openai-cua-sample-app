from computers import Computer
from utils import (
    create_response,
    show_image,
    pp,
    sanitize_message,
    check_blocklisted_url,
)
import json
from typing import Callable


class Agent:
    """
    A sample agent class that can be used to interact with a computer.

    (See simple_cua_loop.py for a simple example without an agent.)
    """

    def __init__(
        self,
        model="computer-use-preview",
        computer: Computer = None,
        tools: list[dict] = [],
        acknowledge_safety_check_callback: Callable = lambda: False,
        brain_file: str = None,
    ):
        self.model = model
        self.computer = computer
        self.tools = tools
        self.print_steps = True
        self.debug = False
        self.show_images = False
        self.acknowledge_safety_check_callback = acknowledge_safety_check_callback

        if computer:
            self.tools += [
                {
                    "type": "computer-preview",
                    "display_width": computer.dimensions[0],
                    "display_height": computer.dimensions[1],
                    "environment": computer.environment,
                },
            ]

        # initialize memory (brain) if provided
        self.brain_file = brain_file
        if self.brain_file:
            # ensure memory file exists
            open(self.brain_file, "a", encoding="utf-8").close()
            memory_tools = [
                {
                    "type": "function",
                    "name": "fetch_memory",
                    "description": "Fetch your memory from the brain file",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                },
                {
                    "type": "function",
                    "name": "write_memory",
                    "description": "Append content to your memory file (Use this to remember important information and context. For example when you create accounts, to save credentials, or to save other information that you should remember.)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Content to append to memory",
                            }
                        },
                        "required": ["content"],
                        "additionalProperties": False,
                    },
                },
            ]
            self.tools += memory_tools
        else:
            self.brain_file = None

    def debug_print(self, *args):
        if self.debug:
            pp(*args)

    def handle_item(self, item):
        """Handle each item; may cause a computer action + screenshot."""
        if item["type"] == "message":
            if self.print_steps:
                print(item["content"][0]["text"])

        if item["type"] == "function_call":
            name, args = item["name"], json.loads(item["arguments"])
            if self.print_steps:
                print(f"{name}({args})")

            # handle memory functions
            if name == "fetch_memory":
                result = self.fetch_memory()
            elif name == "write_memory":
                result = self.write_memory(**args)
            # if function exists on computer, call it
            elif hasattr(self.computer, name):
                method = getattr(self.computer, name)
                method(**args)
                result = "success"
            else:
                # unknown function
                result = None

            return [
                {
                    "type": "function_call_output",
                    "call_id": item["call_id"],
                    "output": result,
                }
            ]

        if item["type"] == "computer_call":
            action = item["action"]
            action_type = action["type"]
            action_args = {k: v for k, v in action.items() if k != "type"}
            if self.print_steps:
                print(f"{action_type}({action_args})")

            method = getattr(self.computer, action_type)
            method(**action_args)

            screenshot_base64 = self.computer.screenshot()
            if self.show_images:
                show_image(screenshot_base64)

            # if user doesn't ack all safety checks exit with error
            pending_checks = item.get("pending_safety_checks", [])
            for check in pending_checks:
                message = check["message"]
                if not self.acknowledge_safety_check_callback(message):
                    raise ValueError(
                        f"Safety check failed: {message}. Cannot continue with unacknowledged safety checks."
                    )

            call_output = {
                "type": "computer_call_output",
                "call_id": item["call_id"],
                "acknowledged_safety_checks": pending_checks,
                "output": {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{screenshot_base64}",
                },
            }

            # additional URL safety checks for browser environments
            if self.computer.environment == "browser":
                current_url = self.computer.get_current_url()
                check_blocklisted_url(current_url)
                call_output["output"]["current_url"] = current_url

            return [call_output]
        return []
    
    def fetch_memory(self) -> str:
        """Read and return all memory from the brain file."""
        if not self.brain_file:
            return ""
        try:
            with open(self.brain_file, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return ""
        except Exception:
            return ""

    def write_memory(self, content: str) -> str:
        """Append content to the brain file memory."""
        if not self.brain_file:
            return ""
        try:
            with open(self.brain_file, "a", encoding="utf-8") as f:
                f.write(f"\n{content}")
        except Exception:
            pass
        return content

    def run_full_turn(
        self, input_items, print_steps=True, debug=False, show_images=False
    ):
        self.print_steps = print_steps
        self.debug = debug
        self.show_images = show_images
        new_items = []

        # keep looping until we get a final response
        while new_items[-1].get("role") != "assistant" if new_items else True:
            self.debug_print([sanitize_message(msg) for msg in input_items + new_items])

            response = create_response(
                model=self.model,
                input=input_items + new_items,
                tools=self.tools,
                truncation="auto",
            )
            self.debug_print(response)

            if "output" not in response and self.debug:
                print(response)
                raise ValueError("No output from model")
            else:
                new_items += response["output"]
                for item in response["output"]:
                    new_items += self.handle_item(item)

        return new_items
