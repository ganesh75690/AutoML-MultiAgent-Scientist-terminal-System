from __future__ import annotations

from dataclasses import dataclass
import os
from ctypes import Structure, byref, c_ulong, c_ulonglong, sizeof, windll
from datetime import datetime
from time import perf_counter, sleep
from typing import Any


class FILETIME(Structure):
    _fields_ = [("dwLowDateTime", c_ulong), ("dwHighDateTime", c_ulong)]


class MEMORYSTATUSEX(Structure):
    _fields_ = [
        ("dwLength", c_ulong),
        ("dwMemoryLoad", c_ulong),
        ("ullTotalPhys", c_ulonglong),
        ("ullAvailPhys", c_ulonglong),
        ("ullTotalPageFile", c_ulonglong),
        ("ullAvailPageFile", c_ulonglong),
        ("ullTotalVirtual", c_ulonglong),
        ("ullAvailVirtual", c_ulonglong),
        ("ullAvailExtendedVirtual", c_ulonglong),
    ]


@dataclass(slots=True)
class SystemStats:
    cpu_usage_percent: float
    memory_usage_percent: float
    total_memory_gb: float
    available_memory_gb: float
    cpu_cores: int
    timestamp: str


def _filetime_to_int(filetime: FILETIME) -> int:
    return (filetime.dwHighDateTime << 32) + filetime.dwLowDateTime


def _get_cpu_times() -> tuple[int, int, int]:
    idle_time = FILETIME()
    kernel_time = FILETIME()
    user_time = FILETIME()
    windll.kernel32.GetSystemTimes(byref(idle_time), byref(kernel_time), byref(user_time))
    return _filetime_to_int(idle_time), _filetime_to_int(kernel_time), _filetime_to_int(user_time)


def _cpu_usage_percent(sample_delay: float = 0.05) -> float:
    try:
        idle_1, kernel_1, user_1 = _get_cpu_times()
        sleep(sample_delay)
        idle_2, kernel_2, user_2 = _get_cpu_times()
        idle_delta = idle_2 - idle_1
        kernel_delta = kernel_2 - kernel_1
        user_delta = user_2 - user_1
        system_delta = kernel_delta + user_delta
        if system_delta <= 0:
            return 0.0
        busy = system_delta - idle_delta
        return max(0.0, min(100.0, (busy / system_delta) * 100.0))
    except Exception:
        return 0.0


def _memory_stats() -> tuple[float, float, float]:
    memory_status = MEMORYSTATUSEX()
    memory_status.dwLength = sizeof(MEMORYSTATUSEX)
    windll.kernel32.GlobalMemoryStatusEx(byref(memory_status))
    total_gb = memory_status.ullTotalPhys / (1024**3)
    available_gb = memory_status.ullAvailPhys / (1024**3)
    usage_percent = float(memory_status.dwMemoryLoad)
    return usage_percent, total_gb, available_gb


def gather_system_stats() -> SystemStats:
    cpu_usage = _cpu_usage_percent()
    memory_usage, total_gb, available_gb = _memory_stats()
    return SystemStats(
        cpu_usage_percent=round(cpu_usage, 1),
        memory_usage_percent=round(memory_usage, 1),
        total_memory_gb=round(total_gb, 2),
        available_memory_gb=round(available_gb, 2),
        cpu_cores=os.cpu_count() or 0,
        timestamp=datetime.utcnow().isoformat(),
    )
