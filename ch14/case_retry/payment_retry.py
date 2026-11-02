"""Starter intenzionalmente incompleto per la mini demo M03."""

from dataclasses import dataclass, field


class TemporaryGatewayError(Exception):
    """Il gateway segnala un errore transitorio."""


@dataclass
class FakeGateway:
    outcomes: list[str]
    attempts: int = 0

    def authorize(self, amount_eur: int, payment_token: str) -> str:
        self.attempts += 1
        outcome = self.outcomes.pop(0)
        if outcome == "transient":
            raise TemporaryGatewayError("gateway temporarily unavailable")
        if outcome == "approved":
            return f"pay-{self.attempts}"
        raise ValueError(f"Unsupported fixture outcome: {outcome}")


@dataclass(frozen=True)
class PaymentResult:
    status: str
    payment_id: str | None
    attempts: int
    message: str


def authorize_with_retry(
    gateway: FakeGateway,
    amount_eur: int,
    payment_token: str,
) -> PaymentResult:
    """TODO: completa i criteri AC-1, AC-2 e AC-3 della specifica."""
    try:
        payment_id = gateway.authorize(amount_eur, payment_token)
    except TemporaryGatewayError:
        return PaymentResult(
            status="FAILED",
            payment_id=None,
            attempts=1,
            message="Pagamento non riuscito",
        )
    return PaymentResult(
        status="APPROVED",
        payment_id=payment_id,
        attempts=1,
        message="Pagamento autorizzato",
    )
