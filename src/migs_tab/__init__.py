"""migs-tab: convert YouTube acoustic guitar tutorials into accurate tabs + tips."""

# Silence noisy import-time warnings from transitive deps before they fire.
import logging
import warnings

logging.getLogger().setLevel(logging.WARNING)
for _name in ("root", "basic_pitch"):
    logging.getLogger(_name).addFilter(
        lambda r: (
            not (
                "tflite-runtime is not installed" in r.getMessage()
                or "Tensorflow is not installed" in r.getMessage()
            )
        )
    )
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*scikit-learn version.*is not supported.*")
warnings.filterwarnings("ignore", message=".*Torch version.*has not been tested with coremltools.*")

__version__ = "0.1.0"
