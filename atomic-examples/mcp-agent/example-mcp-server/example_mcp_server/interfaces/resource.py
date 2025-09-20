"""Interfaces for resource abstractions."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, ClassVar, Type, TypeVar
from pydantic import BaseModel, Field

# Define a type variable for generic model support
T = TypeVar("T", bound=BaseModel)


class BaseResourceInput(BaseModel):
    """Base class for resource input models."""

    model_config = {"extra": "forbid"}  # Equivalent to additionalProperties: false


class ResourceContent(BaseModel):
    """Model for content in resource responses."""

    type: str = Field(default="text", description="Content type identifier")

    # Common fields for all content types
    content_id: Optional[str] = Field(None, description="Optional content identifier")

    # Type-specific fields (using discriminated unions pattern)
    # Text content
    text: Optional[str] = Field(None, description="Text content when type='text'")

    # JSON content (for structured data)
    json_data: Optional[Dict[str, Any]] = Field(None, description="JSON data when type='json'")

    # Model content (will be converted to json_data during serialization)
    model: Optional[Any] = Field(None, exclude=True, description="Pydantic model instance")

    # Resource-specific fields
    uri: Optional[str] = Field(None, description="URI of the resource")
    mime_type: Optional[str] = Field(None, description="MIME type of the resource")

    # Add more content types as needed (e.g., binary, image, etc.)

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization hook to handle model conversion."""
        if self.model and not self.json_data:
            # Convert model to json_data
            if isinstance(self.model, BaseModel):
                self.json_data = self.model.model_dump()
                if not self.type or self.type == "text":
                    self.type = "json"


class ResourceResponse(BaseModel):
    """Model for resource responses."""

    content: List[ResourceContent]

    @classmethod
    def from_model(cls, model: BaseModel) -> "ResourceResponse":
        """Create a ResourceResponse from a Pydantic model.

        This makes it easier to return structured data directly.

        Args:
            model: A Pydantic model instance to convert

        Returns:
            A ResourceResponse with the model data in JSON format
        """
        return cls(content=[ResourceContent(type="json", json_data=model.model_dump(), model=model)])

    @classmethod
    def from_text(cls, text: str, uri: Optional[str] = None, mime_type: Optional[str] = None) -> "ResourceResponse":
        """Create a ResourceResponse from plain text.

        Args:
            text: The text content
            uri: Optional URI of the resource
            mime_type: Optional MIME type

        Returns:
            A ResourceResponse with text content
        """
        return cls(content=[ResourceContent(type="text", text=text, uri=uri, mime_type=mime_type)])


class Resource(ABC):
    """Abstract base class for all resources."""

    name: ClassVar[str]
    description: ClassVar[str]
    uri: ClassVar[str]
    mime_type: ClassVar[Optional[str]] = None
    input_model: ClassVar[Optional[Type[BaseResourceInput]]] = None
    output_model: ClassVar[Optional[Type[BaseModel]]] = None

    @abstractmethod
    async def read(self, input_data: BaseResourceInput) -> ResourceResponse:
        """Execute the resource with given arguments."""
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for the resource."""
        schema = {
            "name": self.name,
            "description": self.description,
            "uri": self.uri,
        }

        if self.mime_type:
            schema["mime_type"] = self.mime_type

        if self.input_model:
            schema["input"] = self.input_model.model_json_schema()

        if self.output_model:
            schema["output"] = self.output_model.model_json_schema()

        return schema
