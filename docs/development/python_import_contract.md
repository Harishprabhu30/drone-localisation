# Python import contract for standalone research scripts

Status: active repository convention.

## Goal

Research scripts under \`scripts/\` must remain directly executable without
depending on another script directory being importable as a Python package.

## Rules

1. Reusable logic belongs under \`src/uavloc/\`.
2. Files under \`scripts/\` are executable entry points, not shared libraries.
3. Do **not** add new imports of the form:

   \`\`\`python
   from scripts.somewhere.other_script import helper
   \`\`\`

   Direct execution changes \`sys.path[0]\` to the script's own directory, so
   repo-root \`scripts\` is not guaranteed to be importable.

4. A standalone script that imports \`uavloc\` should bootstrap the repository
   \`src/\` path from \`__file__\` before its first \`uavloc\` import:

   \`\`\`python
   import sys
   from pathlib import Path

   REPO_ROOT = Path(__file__).resolve().parents[<depth-to-repo>]
   SRC_ROOT = REPO_ROOT / "src"
   if str(SRC_ROOT) not in sys.path:
       sys.path.insert(0, str(SRC_ROOT))

   from uavloc... import ...
   \`\`\`

   The correct parent depth must be determined from the script's actual path,
   not copied blindly.

5. Shell \`PYTHONPATH=$PWD/src\` may still be used for convenience, but scripts
   must not rely on it for correctness when a robust self-bootstrap is easy.

6. New shared utilities should have unit tests under \`tests/\`.

7. For important standalone entry points, add a direct-execution smoke test
   that removes \`PYTHONPATH\` and executes:

   \`\`\`text
   python path/to/script.py --help
   \`\`\`

   This catches import-path regressions without running the experiment.

## RG2.1-B example

\`src/uavloc/evaluation/reference.py\` now owns the reusable prepared-reference
loader. \`rg2_1b_postfreeze_availability_oracle.py\` imports that package helper
after bootstrapping \`repo/src\`. It no longer imports
\`scripts/villoc/retrieval/qv1_3_fixed_budget_multiview_pool.py\`.

Historical scripts are not being rewritten solely for style. Apply this
contract to new code and touch old scripts only when they are actively reused
or maintained.
