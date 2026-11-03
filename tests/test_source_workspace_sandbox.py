import json
import os
from pathlib import Path
import shlex
from tempfile import TemporaryDirectory
from unittest import TestCase, skipUnless
from magents.source_artifacts import source_workspaces


class SandboxWorkspaceTest(TestCase):
    @skipUnless(os.getenv('MAGENTS_SANDBOX_TEST') == '1', 'opt-in official-tool namespace check')
    def test_native_paths_across_files_commands_and_official_tools(self):
        from magents.adapters import WorkspaceTools
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        root = Path(__file__).resolve().parents[1]
        tools_dir = root.parent / 'evolution-lab/official-tool-baseline-v1/runs/qwen35-01/fixed-earth/agent/tools'
        with TemporaryDirectory(dir=root / 'var', prefix='native-mount-check-') as temp:
            source = Path(temp) / 'source/workspace'
            source.mkdir(parents=True)
            (source / 'note.txt').write_text('original')
            (source.parent / 'score.json').write_text('private')
            for name, value in [('bt', 290), ('red', .1), ('nir', .9)]:
                with rasterio.open(source / (name + '.tif'), 'w', driver='GTiff', height=2,
                    width=2, count=1, dtype='float32', transform=from_origin(0, 2, 1, 1)) as out:
                    out.write(np.full((2, 2), value, dtype='float32'), 1)
            diagnostic = Path(temp) / 'diagnostic'
            diagnostic.mkdir()
            config = {'earth_tools': {'tools_dir': str(tools_dir),
                'python': '/srv/user/yangxiao/anaconda3/envs/earth-agent/bin/python',
                'model_results_file': '', 'timeout_seconds': 60}}
            tools = WorkspaceTools(diagnostic, [source], config, official=True, source_workspace=source)
            try:
                self.assertIn('/workspace/note.txt', tools.execute('list_files', {'path': '/workspace'}))
                self.assertEqual(tools.execute('read_file', {'path': '/workspace/note.txt'}), 'original')
                with self.assertRaises(ValueError):
                    tools.box.resolve('/workspace/note.txt', writable=True)
                self.assertEqual(tools.box.resolve('/diagnostics/new.txt', writable=True, allow_missing=True), diagnostic / 'new.txt')
                cmd = "python -c \"from pathlib import Path; print(Path('/workspace/note.txt').read_text()); Path('/diagnostics/new.txt').write_text('new')\""
                result = json.loads(tools.execute('run_command', {'command': cmd}))
                self.assertEqual(result['exit_code'], 0, result)
                self.assertEqual((diagnostic / 'new.txt').read_text(), 'new')
                output = tools.execute('lst_single_channel', {'bt_path': '/workspace/bt.tif',
                    'red_path': '/workspace/red.tif', 'nir_path': '/workspace/nir.tif',
                    'output_path': '/diagnostics/lst.tif'})
                self.assertTrue((diagnostic / 'lst.tif').exists(), output)
                self.assertIn('290', tools.execute('calc_batch_image_mean', {'file_list': ['/workspace/bt.tif']}))
                probe = tools.diagnostic_execute('run_command', {'command': "python -c \"from pathlib import Path; print(Path('/workspace/note.txt').read_text())\""})
                self.assertEqual(json.loads(probe)['exit_code'], 0)
                result = json.loads(tools.execute('run_command', {'command': 'echo changed > /workspace/note.txt'}))
                self.assertNotEqual(result['exit_code'], 0)
                self.assertEqual((source / 'note.txt').read_text(), 'original')
                result = json.loads(tools.execute('run_command', {'command': 'test -e ' + shlex.quote(str(source.parent / 'score.json'))}))
                self.assertNotEqual(result['exit_code'], 0)
            finally:
                tools.close()

    @skipUnless(os.getenv('MAGENTS_SANDBOX_TEST') == '1', 'opt-in namespace check')
    def test_read_only_workspace_and_private_parent(self):
        from rsagent.runtime.sandbox import NamespaceRunner
        root = Path(__file__).resolve().parents[1] / 'var'
        with TemporaryDirectory(dir=root, prefix='mount-check-') as temp:
            run = Path(temp) / 'source'
            source = run / 'workspace'
            source.mkdir(parents=True)
            (source / 'script.py').write_text('original')
            (run / 'score.json').write_text('private')
            diagnostic = Path(temp) / 'diagnostic'
            diagnostic.mkdir()
            mounts, _ = source_workspaces({'s': run})
            code = '''import pathlib,json,sys
p=pathlib.Path(sys.argv[1]); r={"listed":"script.py" in [x.name for x in p.iterdir()],"read":(p/"script.py").read_text(),"host_visible":(p.parent/"score.json").exists()}
try:
 (p/"script.py").write_text("changed")
 r["write_blocked"]=False
except OSError:
 r["write_blocked"]=True
pathlib.Path("/workspace/probe.txt").write_text("diagnostic")
print(json.dumps(r))'''
            result = NamespaceRunner(read_only_paths=mounts).run(diagnostic,
                shlex.join(['python', '-c', code, str(source)]),
                {'PATH': os.environ.get('PATH', '/usr/bin:/bin')}, timeout=30)
            self.assertEqual(result.exit_code, 0, result.stdout)
            self.assertEqual(json.loads(result.stdout), {'listed': True, 'read': 'original',
                'host_visible': False, 'write_blocked': True})
            self.assertEqual((diagnostic / 'probe.txt').read_text(), 'diagnostic')
            self.assertEqual((source / 'script.py').read_text(), 'original')
