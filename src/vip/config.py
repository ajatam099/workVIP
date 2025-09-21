"""Configuration models for the Vision Inspection Pipeline."""


from pydantic import BaseModel, Field


class RunConfig(BaseModel):
    """Configuration for running the defect detection pipeline."""

    input_dir: str = Field(default="input", description="Input directory containing images")
    output_dir: str = Field(default="output", description="Output directory for results")
    defects: list[str] = Field(
        default=["scratches", "contamination", "discoloration", "cracks"],
        description="List of defect types to detect",
    )
    resize_width: int | None = Field(
        default=None, description="Optional width to resize images to"
    )
    save_overlay: bool = Field(default=True, description="Whether to save overlay images")
    save_json: bool = Field(default=True, description="Whether to save JSON results")

    class Config:
        """Pydantic configuration."""

        frozen = True
