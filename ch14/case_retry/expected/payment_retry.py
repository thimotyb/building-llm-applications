"""Implementazione minima di riferimento per la mini demo M03."""

from dataclasses import dataclass


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
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            payment_id = gateway.authorize(amount_eur, payment_token)
        except TemporaryGatewayError:
            if attempt < max_attempts:
                continue
            return PaymentResult(
                status="FAILED",
                payment_id=None,
                attempts=attempt,
                message="Pagamento non riuscito; usa un metodo alternativo",
            )
        return PaymentResult(
            status="APPROVED",
            payment_id=payment_id,
            attempts=attempt,
            message="Pagamento autorizzato",
        )

    raise AssertionError("il ciclo termina sempre con un esito")
