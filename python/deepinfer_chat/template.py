"""
Chat template application

Converts chat messages to prompt text using model-specific templates
"""
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ChatTemplate:
    """Chat template processor"""
    
    def __init__(
        self,
        system_prompt: str = "",
        roles: List[str] = None,
        intra_message_sep: str = "\n",
        inter_message_sep: str = "\n",
    ):
        self.system_prompt = system_prompt
        self.roles = roles or ["user", "assistant"]
        self.intra_message_sep = intra_message_sep
        self.inter_message_sep = inter_message_sep
    
    def apply(self, messages: List[Dict[str, str]]) -> str:
        """
        Apply chat template to messages
        
        Args:
            messages: List of {role: str, content: str} dicts
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # Add system prompt if present
        if self.system_prompt:
            prompt_parts.append(f"system: {self.system_prompt}")
        
        # Add messages
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        
        # Join with inter-message separator
        prompt = self.inter_message_sep.join(prompt_parts)
        
        # Add final role prefix for assistant
        prompt += f"{self.inter_message_sep}assistant:"
        
        return prompt


# Common templates
QWEN_TEMPLATE = ChatTemplate(
    system_prompt="You are a helpful assistant.",
    roles=["user", "assistant"],
    intra_message_sep="\n",
    inter_message_sep="\n\n",
)

LLAMA_TEMPLATE = ChatTemplate(
    system_prompt="You are a helpful, respectful and honest assistant.",
    roles=["user", "assistant"],
    intra_message_sep="\n",
    inter_message_sep="\n\n",
)

DEFAULT_TEMPLATE = ChatTemplate()


def get_template(model_family: str) -> ChatTemplate:
    """Get chat template for a model family"""
    templates = {
        "qwen": QWEN_TEMPLATE,
        "llama": LLAMA_TEMPLATE,
    }
    
    return templates.get(model_family, DEFAULT_TEMPLATE)
