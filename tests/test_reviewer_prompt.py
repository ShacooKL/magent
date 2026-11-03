from unittest import TestCase
from unittest.mock import MagicMock
from magents.engine import Team


class ReviewerPromptTests(TestCase):
    def test_skeptical_review_survives_permission_modes(self):
        for config in ({}, {'official_only': True},
                       {'official_only': True, 'hypothesis_verification': True}):
            with self.subTest(config=config):
                team = Team(None, None, MagicMock(), config)
                prompt = team.prompts['reviewer']
                self.assertIn('NOT established facts', prompt)
                self.assertIn('use evidence_read', prompt)
                self.assertIn('JSON types', prompt)
                self.assertIn('tool_inspect', prompt)
                self.assertIn('smallest discriminating check', prompt)
                self.assertIn('mark the claim unchecked', prompt)
                self.assertNotIn('Act as an independent, skeptical reviewer', team.prompts['executor'])
