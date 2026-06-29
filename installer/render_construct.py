"""Render construct.yaml.j2 -> construct.yaml using values supplied by CI.

Reads BODY_EYE_SYNC_VERSION and LOCAL_CHANNEL from the environment.
"""

import os
from pathlib import Path

from jinja2 import Template

here = Path(__file__).parent
rendered = Template(
    (here / "construct.yaml.j2").read_text(), keep_trailing_newline=True
).render(
    version=os.environ["BODY_EYE_SYNC_VERSION"],
    local_channel=os.environ["LOCAL_CHANNEL"],
)
(here / "construct.yaml").write_text(rendered)
