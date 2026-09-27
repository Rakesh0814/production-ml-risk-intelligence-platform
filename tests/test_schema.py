from app.schemas import CreditRiskRequest


def test_request_defaults_are_valid():
    payload = CreditRiskRequest()
    assert payload.amount > 0
    assert 1 <= payload.status <= 4
