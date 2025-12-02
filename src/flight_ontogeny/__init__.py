"""
flight-ontogeny: Codes for analysing how zebra finch fledglings develop their flight.
"""

from __future__ import annotations

from importlib.metadata import version



__all__ = ("__version__",)


try:
    from importlib.metadata import version
    __version__ = version(__name__)
except Exception:
    __version__ = "0.0.0" 

