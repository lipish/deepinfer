"""
LLM Model Registry

Data structures for LLM model families and specifications.
Simplified version inspired by xinference's model registry.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import json
from pathlib import Path


@dataclass
class LLMPromptStyle:
    """Chat prompt style/template"""
    style_name: str
    system_prompt: str = ""
    roles: List[str] = field(default_factory=lambda: ["user", "assistant"])
    intra_message_sep: str = "\n"
    inter_message_sep: str = "\n"
    stop_token_ids: List[int] = field(default_factory=list)
    stop: List[str] = field(default_factory=list)


@dataclass
class LLMSpecV1:
    """LLM model specification"""
    model_name: str
    model_family: str
    model_format: str  # pytorch, gguf, etc.
    model_size_in_billions: Optional[int] = None
    quantizations: List[str] = field(default_factory=list)
    model_id: Optional[str] = None
    model_revision: Optional[str] = None
    model_hub: str = "huggingface"
    prompt_style: Optional[LLMPromptStyle] = None
    model_description: str = ""
    context_length: Optional[int] = None


@dataclass
class LLMFamilyV2:
    """LLM model family"""
    version: int = 2
    model_type: str = "LLM"
    model_name: str = ""
    model_lang: List[str] = field(default_factory=lambda: ["en"])
    model_ability: List[str] = field(default_factory=lambda: ["generate"])
    model_specs: List[LLMSpecV1] = field(default_factory=list)
    model_family: str = ""
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "version": self.version,
            "model_type": self.model_type,
            "model_name": self.model_name,
            "model_lang": self.model_lang,
            "model_ability": self.model_ability,
            "model_family": self.model_family,
        }


class ModelRegistry:
    """Simple model registry"""
    
    def __init__(self):
        self.families: Dict[str, LLMFamilyV2] = {}
    
    def register_family(self, family: LLMFamilyV2):
        """Register a model family"""
        self.families[family.model_name] = family
    
    def get_family(self, model_name: str) -> Optional[LLMFamilyV2]:
        """Get a model family by name"""
        return self.families.get(model_name)
    
    def list_families(self) -> List[str]:
        """List all registered families"""
        return list(self.families.keys())
    
    @staticmethod
    def load_from_file(path: Path) -> 'ModelRegistry':
        """Load registry from JSON file"""
        registry = ModelRegistry()
        
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                for family_data in data.get("families", []):
                    # TODO: Deserialize LLMFamilyV2 from dict
                    pass
        
        return registry


# Example: Qwen2.5 family
QWEN25_FAMILY = LLMFamilyV2(
    model_name="qwen2.5",
    model_family="qwen",
    model_lang=["en", "zh"],
    model_ability=["generate", "chat"],
    model_specs=[
        LLMSpecV1(
            model_name="Qwen2.5-7B-Instruct",
            model_family="qwen2.5",
            model_format="pytorch",
            model_size_in_billions=7,
            model_id="Qwen/Qwen2.5-7B-Instruct",
            context_length=32768,
            prompt_style=LLMPromptStyle(
                style_name="qwen",
                system_prompt="You are a helpful assistant.",
                roles=["user", "assistant"],
            )
        )
    ]
)
