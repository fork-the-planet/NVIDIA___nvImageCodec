#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Rewrite recipe/meta.yaml: set version and source for local or Docker build.
# Local builds use `source: path:` to capture the working tree (including
# uncommitted changes); Docker builds use `source: git_url:` against the
# mounted source repo.

import os
import re
import sys


def main():
    meta_yaml = os.environ.get("META_YAML")
    detected_version = os.environ.get("DETECTED_VERSION")
    if not meta_yaml:
        print("rewrite_meta_yaml.py: error: META_YAML environment variable is required but not set.", file=sys.stderr)
        sys.exit(1)
    if not detected_version:
        print("rewrite_meta_yaml.py: error: DETECTED_VERSION environment variable is required but not set.", file=sys.stderr)
        sys.exit(1)

    build_local = os.environ.get("CONDA_BUILD_LOCAL", "0") == "1"
    project_root = os.environ.get("PROJECT_ROOT", "")

    if build_local:
        if not project_root:
            print("rewrite_meta_yaml.py: error: PROJECT_ROOT environment variable is required for local builds but not set.", file=sys.stderr)
            sys.exit(1)
        source_key = "path"
        source_value = project_root
    else:
        source_key = "git_url"
        source_value = "/home/conda/source_repo"

    with open(meta_yaml, "r") as f:
        content = f.read()

    version_pattern = r'{%\s*set\s+version\s*=\s*"[^"]+"\s*%}'
    version_replacement = f'{{% set version = "{detected_version}" %}}'
    content, version_subs = re.subn(version_pattern, version_replacement, content)
    if version_subs == 0:
        msg = (
            "rewrite_meta_yaml.py: error: version line not found in meta.yaml; "
            f"expected a line matching {version_pattern!r}. "
            f"Attempted value: detected_version={detected_version!r}."
        )
        print(msg, file=sys.stderr)
        sys.exit(1)

    source_pattern = r"source:[ \t]*\n(?:[ \t]+(?:url|sha256|git_url|path):[^\n]*\n)+"
    source_replacement = f"source:\n  {source_key}: {source_value}\n"
    content, source_subs = re.subn(source_pattern, source_replacement, content)
    if source_subs == 0:
        msg = (
            "rewrite_meta_yaml.py: error: source block not found in meta.yaml; "
            f"expected a block matching {source_pattern!r}. "
            f"Attempted value: {source_key}={source_value!r}."
        )
        print(msg, file=sys.stderr)
        sys.exit(1)

    with open(meta_yaml, "w") as f:
        f.write(content)

    print(f"✓ Modified meta.yaml: version={detected_version}, {source_key}={source_value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
