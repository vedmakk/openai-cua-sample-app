from typing import List, Dict, Any
from .memory_provider import MemoryProvider

class FileMemoryProvider(MemoryProvider):
    """
    A simple file-backed memory provider.
    """
    def __init__(self, memory_file: str):
        self.memory_file = memory_file
        # ensure the memory file exists
        open(self.memory_file, 'a', encoding='utf-8').close()

    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": "fetch_memory",
                "description": "Fetch the agent's memory from the memory file",
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
                "description": "Append content to the agent's memory file",
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

    def handle_call(self, name: str, arguments: Dict[str, Any]) -> Any:
        if name == "fetch_memory":
            try:
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception:
                return ""
        elif name == "write_memory":
            content = arguments.get("content", "")
            try:
                with open(self.memory_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n{content}")
            except Exception:
                pass
            return content
        else:
            raise ValueError(f"MemoryProvider cannot handle function '{name}'")