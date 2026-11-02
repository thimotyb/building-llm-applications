import unittest

from payment_retry import FakeGateway, authorize_with_retry


class PaymentRetryTests(unittest.TestCase):
    def test_approval_on_first_attempt_stops(self):
        gateway = FakeGateway(["approved"])

        result = authorize_with_retry(gateway, 2000, "tok-demo")

        self.assertEqual(result.status, "APPROVED")
        self.assertEqual(result.payment_id, "pay-1")
        self.assertEqual(result.attempts, 1)
        self.assertEqual(gateway.attempts, 1)

    def test_transient_then_approval_retries_and_stops(self):
        gateway = FakeGateway(["transient", "approved"])

        result = authorize_with_retry(gateway, 2000, "tok-demo")

        self.assertEqual(result.status, "APPROVED")
        self.assertEqual(result.payment_id, "pay-2")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(gateway.attempts, 2)

    def test_three_transient_errors_suggest_alternative(self):
        gateway = FakeGateway(["transient", "transient", "transient"])

        result = authorize_with_retry(gateway, 2000, "tok-demo")

        self.assertEqual(result.status, "FAILED")
        self.assertIsNone(result.payment_id)
        self.assertEqual(result.attempts, 3)
        self.assertEqual(gateway.attempts, 3)
        self.assertIn("alternativo", result.message.lower())


if __name__ == "__main__":
    unittest.main()
