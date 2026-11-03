import unittest
from experiments.earthbench_q1 import submitted_choice


class ScoringTests(unittest.TestCase):
    def test_structured_choice(self):
        self.assertEqual(submitted_choice({"final_submission": {"status": "completed", "answer": "B"}}), "B")

    def test_structured_labeled_option(self):
        self.assertEqual(submitted_choice({"final_submission": {"status": "completed", "answer": "C. 28.07%"}}), "C")
        self.assertIsNone(submitted_choice({"final_submission": {"status": "completed", "answer": "Could be C"}}))

    def test_incomplete_is_not_scored_as_an_answer(self):
        self.assertIsNone(submitted_choice({"answer": "<Answer>B</Answer>",
            "final_submission": {"status": "incomplete", "answer": "B"}}))

    def test_legacy_tag(self):
        self.assertEqual(submitted_choice({"answer": "<Answer>B</Answer>"}), "B")
        self.assertIsNone(submitted_choice({"answer": "I will inspect B"}))
