from pathlib import Path
from unittest import TestCase
from experiments.paired_controls import job_command


class PairedCommandsTests(TestCase):
    def test_retrieval_is_the_only_permission_difference(self):
        for condition in ('with-skill', 'without-skill'):
            command = job_command({'task': 93, 'condition': condition, 'directory': condition}, Path('/tmp/test'), None)
            self.assertIn('--official-only', command)
            self.assertEqual('--no-skill-retrieval' in command, condition == 'without-skill')
            self.assertNotIn('--skill-file', command)

    def test_old_permission_comparison_stays_skill_free(self):
        for condition in ('auxiliary', 'official-only'):
            command = job_command({'task': 4, 'condition': condition, 'directory': condition}, Path('/tmp/test'), None)
            self.assertIn('--no-skill-retrieval', command)
            self.assertEqual('--official-only' in command, condition == 'official-only')
