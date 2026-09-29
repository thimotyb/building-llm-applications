"""Small offline tests for the deterministic flight-upgrade policy."""

import unittest

from .agent import miles_route, parse_miles, root_agent


class UpgradeWorkflowTests(unittest.TestCase):
    def test_parse_miles(self):
        self.assertEqual(parse_miles("Member has 12,000 miles"), 12_000)

    def test_three_routes_and_boundaries(self):
        cases = {
            4_999: "DENY",
            5_000: "GET_CONSENT",
            20_000: "GET_CONSENT",
            20_001: "AUTO_APPROVE",
        }
        for miles, expected in cases.items():
            with self.subTest(miles=miles):
                self.assertEqual(miles_route(miles), expected)

    def test_workflow_builds(self):
        self.assertEqual(root_agent.name, "flight_upgrade_workflow")


if __name__ == "__main__":
    unittest.main()
