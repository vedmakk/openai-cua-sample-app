from abc import ABC, abstractmethod
from typing import List, Dict, Any


class MemoryProvider(ABC):
    """
    Abstract base class for memory providers.
    Memory providers can supply tool schemas and handle function calls for memory operations.
    """
    @abstractmethod
    def get_tools(self) -> List[Dict[str, Any]]:
        """Return a list of tool schemas for this memory provider."""
        pass

    @abstractmethod
    def handle_call(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Handle a function call with the given name and arguments."""
        pass