"""Compatibilite locale pour MLflow UI avec Python 3.14.

MLflow 3.14 importe encore ``Traversable`` depuis ``importlib.abc`` dans son
serveur UI. Avec Python 3.14, ce symbole est expose via ``importlib.resources.abc``.
Ce shim garde le tracking applicatif intact et permet au sous-process uvicorn de
demarrer l'UI MLflow locale.
"""

from __future__ import annotations

import importlib.abc
import importlib.resources.abc

if not hasattr(importlib.abc, "Traversable"):
    importlib.abc.Traversable = importlib.resources.abc.Traversable
