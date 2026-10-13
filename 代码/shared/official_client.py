"""Official four-endpoint HTTP client shared by both questions."""

from __future__ import annotations

import csv
import json
import logging
import math
import socket
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


class CommunicationError(RuntimeError):
    pass


class RequestRejected(CommunicationError):
    pass


@dataclass(slots=True)
class RetryPolicy:
    attempts: int = 5
    timeout_s: float = 5.0
    initial_backoff_s: float = 0.2
    max_backoff_s: float = 2.0


class SimulatorClient:
    """Official API transport; accepts either question's point type."""

    def __init__(self, robot_id: str, base_url: str = "http://127.0.0.1:2026",
                 arena_id: str = "default", retry: RetryPolicy | None = None,
                 log_path: str | Path = "robot_dog.jsonl",
                 summary_stem: str | None = None) -> None:
        if not robot_id or len(robot_id.encode("utf-8")) > 64:
            raise ValueError("robot_id must contain 1..64 UTF-8 bytes")
        self.robot_id = robot_id
        self.base_url = self._normalize_base_url(base_url)
        self.arena_id = arena_id
        self.retry = retry or RetryPolicy()
        self._counter = 0
        self.virtual_time_s = 0.0
        self.position: Any = None
        self.current_channel = 1
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_csv_path = self.log_path.with_name(
            f"{summary_stem or self.log_path.stem}_运行指标.csv"
        )
        self._session_start: float | None = None
        self._enter_real_ms: float | None = None
        self._enter_virtual_s = 0.0
        self._case_code = ""
        self._cleared_channels: set[int] = set()
        self.logger = logging.getLogger("official_simulator.communication")

    def _request_id(self, action: str) -> str:
        self._counter += 1
        return f"{action}-{self._counter}-{uuid.uuid4().hex[:10]}"

    def _base(self, request_id: str) -> dict[str, Any]:
        return {"arena_id": self.arena_id, "robot_id": self.robot_id, "request_id": request_id}

    def _write_log(self, record: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as stream:
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            stream.write(line.replace(self.robot_id, "[REDACTED]") + "\n")

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if path not in {"/enter", "/measure", "/clear", "/exit"}:
            raise ValueError(f"unsupported simulator path: {path!r}")
        request_url = self.base_url + path
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        backoff = self.retry.initial_backoff_s
        last_error: Exception | None = None
        for attempt in range(1, self.retry.attempts + 1):
            try:
                request = Request(request_url, data=body, headers={"Content-Type": "application/json"}, method="POST")
                with urlopen(request, timeout=self.retry.timeout_s) as http_res:
                    status = http_res.status
                    response = json.loads(http_res.read().decode("utf-8"))
                self._write_log({"url": request_url, "path": path, "request": payload, "status": status, "response": response})
                if status != 200:
                    raise CommunicationError(f"{path}: unexpected HTTP {status}")
                if response.get("accepted") is not True:
                    raise RequestRejected(f"{path}: request rejected: {response}")
                self.virtual_time_s = float(response.get("virtual_time_s", self.virtual_time_s))
                return response
            except HTTPError as exc:
                raw = exc.read().decode("utf-8", errors="replace")
                try:
                    response = json.loads(raw)
                except json.JSONDecodeError:
                    response = {"raw": raw}
                self._write_log({"url": request_url, "path": path, "request": payload, "status": exc.code, "response": response})
                retryable = exc.code == 429 or 500 <= exc.code <= 599
                if retryable and attempt < self.retry.attempts:
                    retry_after = exc.headers.get("Retry-After")
                    try:
                        wait_s = float(retry_after) if retry_after is not None else backoff
                    except ValueError:
                        wait_s = backoff
                    time.sleep(max(0.0, min(wait_s, self.retry.max_backoff_s)))
                    backoff = min(self.retry.max_backoff_s, backoff * 2)
                    continue
                raise CommunicationError(f"{path}: HTTP {exc.code}: {response}") from exc
            except (URLError, TimeoutError, socket.timeout, ConnectionError, json.JSONDecodeError) as exc:
                last_error = exc
                self.logger.warning("%s attempt %d/%d failed: %s", path, attempt, self.retry.attempts, exc)
                if attempt < self.retry.attempts:
                    time.sleep(backoff)
                    backoff = min(self.retry.max_backoff_s, backoff * 2)
        raise CommunicationError(f"{path}: no valid response after retries") from last_error

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        parsed = urlsplit(base_url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url 必须类似 http://127.0.0.1:2026，不能省略 http://")
        if parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
            raise ValueError("base_url 只能填写服务器地址和端口，不能包含 /enter、/measure 等路径")
        return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))

    def enter(self) -> dict[str, Any]:
        response = self._post("/enter", self._base(self._request_id("enter")))
        self._session_start = time.monotonic()
        self._enter_real_ms = response.get("real_timestamp_ms")
        self._enter_virtual_s = float(response.get("virtual_time_s", 0.0))
        self._case_code = str(response.get("test_case_code") or response.get("case_code") or "")
        self._cleared_channels.clear()
        return response

    def measure(self, position: Any, channel: int) -> dict[str, Any]:
        self._validate_action(position, channel)
        payload = self._base(self._request_id("measure")) | {"position": {"x": position.x, "y": position.y}, "channel": channel}
        response = self._post("/measure", payload)
        self.position, self.current_channel = position, channel
        return response

    def clear(self, position: Any, channel: int) -> dict[str, Any]:
        self._validate_action(position, channel)
        payload = self._base(self._request_id("clear")) | {"position": {"x": position.x, "y": position.y}, "channel": channel}
        response = self._post("/clear", payload)
        self.position = position
        if response.get("clear_result") == "success":
            self._cleared_channels.add(channel)
        return response

    def exit(self) -> dict[str, Any]:
        response = self._post("/exit", self._base(self._request_id("exit")))
        if self._session_start is not None:
            try:
                self._write_summary_csv(response)
            except OSError:
                self.logger.exception("could not write session metrics CSV")
            self._session_start = None
        return response

    def _write_summary_csv(self, exit_response: dict[str, Any]) -> None:
        case_code = str(exit_response.get("test_case_code") or exit_response.get("case_code") or self._case_code)
        case_code = case_code.replace(self.robot_id, "[REDACTED]")
        count = len(self._cleared_channels)
        virtual_s = max(0.0, float(exit_response.get("virtual_time_s", self.virtual_time_s)) - self._enter_virtual_s)
        average_s = f"{virtual_s / count:.2f}" if count else ""
        end_ms = exit_response.get("real_timestamp_ms")
        if self._enter_real_ms is not None and end_ms is not None:
            program_s = max(0.0, (float(end_ms) - float(self._enter_real_ms)) / 1000.0)
        else:
            program_s = max(0.0, time.monotonic() - self._session_start)
        new_file = not self.summary_csv_path.exists() or self.summary_csv_path.stat().st_size == 0
        with self.summary_csv_path.open("a", encoding="utf-8-sig" if new_file else "utf-8", newline="") as stream:
            writer = csv.writer(stream)
            if new_file:
                writer.writerow(["测试案例编码", "清除干扰源个数", "平均定位清除时间", "程序运行时间"])
            writer.writerow([case_code, count, average_s, f"{program_s:.2f}"])

    @staticmethod
    def _validate_action(position: Any, channel: int) -> None:
        if not isinstance(channel, int) or isinstance(channel, bool) or not (1 <= channel <= 20):
            raise ValueError("channel must be an integer in 1..20")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) and abs(v) <= 2_000_000 for v in (position.x, position.y)):
            raise ValueError("position components must be finite and within +/-2,000,000")
