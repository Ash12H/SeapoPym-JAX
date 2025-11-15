"""Transport module: Centralized advection and diffusion for distributed simulation.

This module provides the TransportWorker class, a Ray remote actor that handles
all transport operations on the global domain using JAX-optimized algorithms.
"""

from seapopym_message.transport.worker import TransportWorker

__all__ = ["TransportWorker"]
