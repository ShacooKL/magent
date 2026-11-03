"""Expose source workspaces read-only, without their parent run records."""
from pathlib import Path


def source_workspaces(sources, allowed_external_roots=()):
    mounts, mapping = [], {}
    external = [Path(p).resolve() for p in allowed_external_roots]
    for name, directory in sources.items():
        workspace = Path(directory) / 'workspace'
        if workspace.is_symlink():
            raise ValueError(f'Source workspace must not be a symlink: {workspace}')
        if not workspace.exists():
            mapping[name] = {'original_path': '/workspace', 'read_only_path': None, 'status': 'missing'}
            continue
        if not workspace.is_dir():
            raise ValueError(f'Source workspace is not a directory: {workspace}')
        workspace = workspace.resolve()
        for path in workspace.rglob('*'):
            if path.is_symlink() and not any(path.resolve().is_relative_to(root) for root in [workspace, *external]):
                raise ValueError(f'Source link escapes approved inputs: {path}')
        mounts.append(workspace)
        mapping[name] = {'original_path': '/workspace', 'read_only_path': str(workspace),
                         'status': 'available', 'snapshot': 'end_of_run'}
    return mounts, mapping
