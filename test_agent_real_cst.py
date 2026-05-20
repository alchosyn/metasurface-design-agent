"""Integration test: directly invoke the agent's run_cst tool multiple times.

Simulates exactly what the LLM agent does for several geometries in a row,
bypassing the LLM. This catches multi-call COM state issues.
"""
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

from agent.tools import SharedState, build_tools
from agent.cst.connection import CSTConnection

PROJECT_PATH = r"D:\alchosyn\01_dissertation\unit_cell\final.cst"

shared = SharedState(max_cst_calls=20, mock_cst=False)
shared.cst_connection = CSTConnection(PROJECT_PATH)
shared.cst_project_path = PROJECT_PATH

tools = build_tools(shared)
run_cst = next(t for t in tools if t.name == "run_cst")

# 5 diverse points — must all produce different results
test_points = [
    {"L": 80.0, "W": 60.0, "h": 620.0},
    {"L": 200.0, "W": 100.0, "h": 700.0},
    {"L": 320.0, "W": 60.0, "h": 840.0},
    {"L": 250.0, "W": 150.0, "h": 750.0},
    {"L": 180.0, "W": 80.0, "h": 680.0},
]

for i, point in enumerate(test_points, 1):
    print(f"\n{'='*60}\n[{i}/{len(test_points)}] run_cst({point})\n{'='*60}")
    result = run_cst.invoke(point)
    print(result)

shared.cst_connection.close()

print(f"\n\n=== Final database ({len(shared.cst_database)}/{len(test_points)} entries) ===")
for r in shared.cst_database:
    print(f"  L={r['L']:.0f}, W={r['W']:.0f}, h={r['h']:.0f}: "
          f"dphi={r['dphi_deg']:.1f}, T_avg={r['T_avg_pct']:.1f}%")

if len(shared.cst_database) == len(test_points):
    print("\nSUCCESS: all simulations completed.")
else:
    print(f"\nFAILED: only {len(shared.cst_database)}/{len(test_points)} succeeded.")
    sys.exit(1)
