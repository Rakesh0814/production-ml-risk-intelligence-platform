from pydantic import BaseModel, Field


class CreditRiskRequest(BaseModel):
    status: int = Field(3, ge=1, le=4)
    duration: int = Field(24, ge=1, le=72)
    credit_history: int = Field(3, ge=0, le=4)
    purpose: int = Field(3, ge=0, le=10)
    amount: float = Field(3000, gt=0)
    savings: int = Field(3, ge=1, le=5)
    employment_duration: int = Field(3, ge=1, le=5)
    installment_rate: int = Field(3, ge=1, le=4)

    other_debtors: int = Field(1, ge=1, le=3)
    present_residence: int = Field(3, ge=1, le=4)
    property: int = Field(3, ge=1, le=4)
    other_installment_plans: int = Field(3, ge=1, le=3)
    housing: int = Field(3, ge=1, le=3)
    number_credits: int = Field(1, ge=1, le=4)
    job: int = Field(3, ge=1, le=4)
    people_liable: int = Field(2, ge=1, le=2)
    telephone: int = Field(2, ge=1, le=2)
