"""migs-tab: convert YouTube acoustic guitar tutorials into accurate tabs + tips."""

# Silence noisy import-time warnings from transitive deps before they fire.
# coremltools and basic-pitch print directly to stderr at module-load
# time (UserWarnings raised before the warnings filter would catch them,
# plus raw print() statements). We pre-import the noisy modules here
# under a redirected stderr so the CLI's actual output stays clean.

import contextlib
import io
import logging
import warnings

# Filter via warnings module (catches anything that goes through warnings).
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*scikit-learn version.*is not supported.*")
warnings.filterwarnings("ignore", message=".*Torch version.*has not been tested with coremltools.*")

# Filter logging-level noise from basic_pitch.
logging.getLogger().setLevel(logging.WARNING)
for _name in ("root", "basic_pitch"):
    logging.getLogger(_name).addFilter(
        lambda r: (
            not (
                "tflite-runtime is not installed" in r.getMessage()
                or "Tensorflow is not installed" in r.getMessage()
                or "scikit-learn version" in r.getMessage()
                or "Torch version" in r.getMessage()
            )
        )
    )

# coremltools writes to stderr via raw print() inside its module __init__,
# bypassing the warnings system. Pre-import the offender with stderr muted
# so the warning fires once during this module's load and never again.
_devnull = io.StringIO()
with contextlib.redirect_stderr(_devnull):
    try:
        import coremltools  # noqa: F401
    except ImportError:
        pass

__version__ = "0.1.0"
