from pydantic import BaseModel, Field, field_validator
class Shipment(BaseModel):
    units: int = Field(gt=0, le=12)
    destination: str
    retries: int = Field(default="3", validate_default=True)
    @field_validator("destination", mode="before")
    @classmethod
    def trim_destination(cls, value):
        return value.strip() if isinstance(value, str) else value
ACCEPTED_SAMPLE = {"units": 12, "destination": " Busan "}
REJECTED_SAMPLE = {"units": 0, "destination": "Seoul"}
