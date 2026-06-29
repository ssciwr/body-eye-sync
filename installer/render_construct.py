"""Render construct.yaml.j2 -> construct.yaml using the version supplied by CI.

We render the template ourselves with a real Jinja engine rather than relying on
constructor's built-in templating, which only fires when the file is otherwise
invalid YAML.

Reads BODY_EYE_SYNC_VERSION from the environment.
"""

import os
from pathlib import Path

from jinja2 import Template

here = Path(__file__).parent
rendered = Template(
    (here / "construct.yaml.j2").read_text(), keep_trailing_newline=True
).render(version=os.environ["BODY_EYE_SYNC_VERSION"])
(here / "construct.yaml").write_text(rendered)
