"""System telemetry sampling via ``psutil``.

``TelemetryMonitor.sample()`` returns a :class:`~app.models.schemas.Telemetry`
snapshot. Network throughput (``net_up`` / ``net_down``) is reported as an
instantaneous rate in **KB/s**, computed from the delta between this sample's
cumulative byte counters and the previous sample's -- never as a raw
cumulative total, since a growing total is not useful telemetry to stream to
a HUD every second.

Every individual sensor read is wrapped so a platform missing that sensor
(no battery, no exposed CPU temperature, a sandboxed container with a
restricted ``/proc``) degrades that one field to ``None``/``0`` instead of
raising out of ``sample()``. ``psutil`` itself is a hard dependency of this
module (it is listed in ``requirements.txt``), but every *individual call*
into it is still defensively wrapped, since ``psutil``'s per-platform sensor
coverage is inconsistent (e.g. ``sensors_temperatures`` is Linux-only and
``sensors_battery`` is desktop-only).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from app.models.schemas import Telemetry

logger = logging.getLogger(__name__)

_BYTES_PER_KB = 1024.0


@dataclass
class _NetSnapshot:
    bytes_sent: int
    bytes_recv: int
    timestamp: float


class TelemetryMonitor:
    """Stateful sampler: tracks the previous network counters to derive rates.

    One instance is expected to live for the lifetime of a connection /
    process (see ``run``), since the very first ``sample()`` has no prior
    reading to diff against and reports ``net_up``/``net_down`` as ``0.0``.
    """

    def __init__(self) -> None:
        self._prev_net: _NetSnapshot | None = None

    def sample(self) -> Telemetry:
        """Take one point-in-time telemetry reading. Never raises."""
        cpu = self._read_cpu_percent()
        mem = self._read_mem_percent()
        disk = self._read_disk_percent()
        net_up, net_down = self._read_net_rates()
        battery = self._read_battery_percent()
        uptime_s = self._read_uptime_s()
        processes = self._read_process_count()
        temp_c = self._read_cpu_temp()

        return Telemetry(
            cpu=cpu,
            mem=mem,
            disk=disk,
            net_up=net_up,
            net_down=net_down,
            battery=battery,
            uptime_s=uptime_s,
            processes=processes,
            temp_c=temp_c,
        )

    async def run(
        self,
        on_sample: Callable[[Telemetry], Awaitable[None]],
        interval_ms: int = 1000,
    ) -> None:
        """Broadcast loop: sample on a fixed interval and await ``on_sample``.

        Runs until cancelled (the normal way callers stop it -- e.g. a
        WebSocket disconnect cancelling the task). A failure in ``on_sample``
        (e.g. a closed socket) is logged and the loop continues rather than
        dying silently; a failure in ``sample()`` itself cannot happen by
        contract, but is guarded anyway for defense in depth.
        """
        interval_s = max(0.05, interval_ms / 1000.0)
        while True:
            try:
                telemetry = self.sample()
                await on_sample(telemetry)
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - defensive, caller-supplied callback
                logger.warning("Telemetry broadcast callback raised; continuing loop", exc_info=True)
            await asyncio.sleep(interval_s)

    # -- individual sensor reads, each independently fail-safe -------------

    @staticmethod
    def _read_cpu_percent() -> float:
        try:
            import psutil

            # interval=None -> non-blocking, compares against the last call
            # (0.0 on this process's very first call, which is fine: 0..100).
            return float(psutil.cpu_percent(interval=None))
        except Exception:
            logger.warning("Failed to read CPU percent", exc_info=True)
            return 0.0

    @staticmethod
    def _read_mem_percent() -> float:
        try:
            import psutil

            return float(psutil.virtual_memory().percent)
        except Exception:
            logger.warning("Failed to read memory percent", exc_info=True)
            return 0.0

    @staticmethod
    def _read_disk_percent() -> float:
        try:
            import psutil

            return float(psutil.disk_usage("/").percent)
        except Exception:
            logger.warning("Failed to read disk usage", exc_info=True)
            return 0.0

    def _read_net_rates(self) -> tuple[float, float]:
        """Return ``(upload_kb_s, download_kb_s)`` computed from counter deltas."""
        try:
            import psutil

            counters = psutil.net_io_counters()
            now = time.monotonic()
            current = _NetSnapshot(
                bytes_sent=counters.bytes_sent, bytes_recv=counters.bytes_recv, timestamp=now
            )

            previous = self._prev_net
            self._prev_net = current

            if previous is None:
                return 0.0, 0.0

            elapsed = current.timestamp - previous.timestamp
            if elapsed <= 0:
                return 0.0, 0.0

            sent_delta = max(0, current.bytes_sent - previous.bytes_sent)
            recv_delta = max(0, current.bytes_recv - previous.bytes_recv)

            net_up = (sent_delta / _BYTES_PER_KB) / elapsed
            net_down = (recv_delta / _BYTES_PER_KB) / elapsed
            return round(net_up, 2), round(net_down, 2)
        except Exception:
            logger.warning("Failed to read network counters", exc_info=True)
            return 0.0, 0.0

    @staticmethod
    def _read_battery_percent() -> float | None:
        try:
            import psutil

            battery = psutil.sensors_battery()
            if battery is None:
                return None
            return float(battery.percent)
        except Exception:
            # AttributeError on platforms without sensors_battery at all.
            logger.debug("Battery sensor unavailable on this platform", exc_info=True)
            return None

    @staticmethod
    def _read_uptime_s() -> int:
        try:
            import psutil

            return max(0, int(time.time() - psutil.boot_time()))
        except Exception:
            logger.warning("Failed to read system uptime", exc_info=True)
            return 0

    @staticmethod
    def _read_process_count() -> int:
        try:
            import psutil

            return len(psutil.pids())
        except Exception:
            logger.warning("Failed to read process count", exc_info=True)
            return 0

    @staticmethod
    def _read_cpu_temp() -> float | None:
        try:
            import psutil

            temps = psutil.sensors_temperatures()
            if not temps:
                return None
            for readings in temps.values():
                if readings:
                    return float(readings[0].current)
            return None
        except Exception:
            # AttributeError on platforms without sensors_temperatures at all
            # (notably macOS), or PermissionError in some sandboxes.
            logger.debug("CPU temperature sensor unavailable on this platform", exc_info=True)
            return None
