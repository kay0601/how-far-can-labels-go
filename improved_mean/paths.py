"""Where the scripts find the training data, the official tools and a team's own scoring logs.

Each location can be overridden with an environment variable:

    PAIEC_BASELINE  the official baseline repository, after tools/prepare_data.py has written
                    payload/data and matharena has been fetched into payload_extra
                    (default: paiec_baseline/ next to this folder)
    PAIEC_TOOLS     the starting kit's tools/ folder, used by the end-to-end tests
                    (default: tools/ next to this folder)
    PAIEC_FEEDBACK  Codabench scoring logs, saved as <submission id>_scoring_stdout.txt
                    (default: feedback/ in this folder)
    PAIEC_RESULTS   where figure_data.py and proxy_robustness.py write their JSON output
                    (default: report/figures/ next to this folder if it exists, else results/
                    in this folder)
"""

import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.environ.get("PAIEC_BASELINE", os.path.join(HERE, os.pardir, "paiec_baseline"))
# matharena is excluded by prepare_data.py (response_type "mixed") but is binary apart from
# about 60 rows, which the loaders drop; it is fetched separately into payload_extra.
DATA_DIRS = [os.path.join(BASELINE, "payload", "data"), os.path.join(BASELINE, "payload_extra")]
TOOLS = os.environ.get("PAIEC_TOOLS", os.path.join(HERE, os.pardir, "tools"))
FEEDBACK = os.environ.get("PAIEC_FEEDBACK", os.path.join(HERE, "feedback"))
_FIGURES = os.path.join(HERE, os.pardir, "report", "figures")
RESULTS = os.environ.get("PAIEC_RESULTS", _FIGURES if os.path.isdir(_FIGURES) else os.path.join(HERE, "results"))
