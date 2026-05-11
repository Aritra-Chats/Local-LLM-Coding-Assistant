"""connectivity.py — Internet connectivity probe for Sentinel startup.

Used by SentinelRuntime.initialise() (Section 3) to determine whether
online mode is available before presenting the mode-selection prompt.

Usage::

    from core.connectivity import ConnectivityChecker
    available = ConnectivityChecker.check()
"""
from __future__ import annotations

import socket


class ConnectivityChecker:
    """Probes a set of well-known hosts to detect internet connectivity.

    All methods are static — no instance state is required.
    """

    PROBE_HOSTS = [
        ("8.8.8.8",    53),    # Google Public DNS
        ("1.1.1.1",    53),    # Cloudflare DNS
        ("ollama.com", 443),   # Ollama Cloud endpoint
    ]
    TIMEOUT_SECONDS = 3

    @staticmethod
    def check() -> bool:
        """Return True if at least one probe host is reachable.

        Iterates :attr:`PROBE_HOSTS` and attempts a TCP connection to each.
        Returns on the first successful connection.

        Returns:
            ``True`` if the internet is reachable, ``False`` otherwise.
        """
        for host, port in ConnectivityChecker.PROBE_HOSTS:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(ConnectivityChecker.TIMEOUT_SECONDS)
                sock.connect((host, port))
                sock.close()
                return True
            except (socket.error, OSError):
                continue
        return False

    @staticmethod
    def check_ollama_cloud() -> bool:
        """Return True specifically if ollama.com:443 is reachable.

        Returns:
            ``True`` if Ollama Cloud is reachable, ``False`` otherwise.
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(ConnectivityChecker.TIMEOUT_SECONDS)
            sock.connect(("ollama.com", 443))
            sock.close()
            return True
        except (socket.error, OSError):
            return False
