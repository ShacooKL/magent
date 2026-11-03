from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from magents.source_artifacts import source_workspaces


class ArtifactTests(TestCase):
    def test_whole_workspace_not_parent_and_explicit_mapping(self):
        with TemporaryDirectory() as temp:
            run = Path(temp) / 'run'
            workspace = run / 'workspace'
            workspace.mkdir(parents=True)
            (workspace / 'lst.tif').touch()
            (workspace / 'answer.json').write_text('private')
            (run / 'state.json').write_text('private')
            (workspace / 'linked.tif').symlink_to(workspace / 'lst.tif')
            mounts, mapping = source_workspaces({'source-1': run, 'missing': Path(temp) / 'absent'})
            self.assertEqual(mounts, [workspace])
            self.assertEqual(mapping['source-1']['original_path'], '/workspace')
            self.assertEqual(mapping['source-1']['read_only_path'], str(workspace))
            self.assertEqual(mapping['missing']['status'], 'missing')
            (workspace / 'external').symlink_to(run / 'state.json')
            with self.assertRaisesRegex(ValueError, 'escapes'):
                source_workspaces({'s': run})
