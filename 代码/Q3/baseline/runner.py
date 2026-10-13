from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from common.communication import RetryPolicy, SimulatorClient
from baseline.controller import RobotController


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CUMCM B modular robot-dog solver")
    parser.add_argument("--mode", choices=("q3",), default="q3")
    parser.add_argument("--robot-id", default=os.getenv("ROBOT_ID"))
    parser.add_argument("--base-url", default=os.getenv("SIMULATOR_URL", "http://127.0.0.1:2026"))
    parser.add_argument("--log", type=Path, default=Path("logs/robot_dog.jsonl"))
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--retries", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.robot_id:
        args.robot_id = input("请输入当前模拟器登录的参赛队号：").strip()
    if not args.robot_id:
        raise SystemExit("参赛队号不能为空")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    client = SimulatorClient(
        robot_id=args.robot_id,
        base_url=args.base_url,
        retry=RetryPolicy(attempts=args.retries, timeout_s=args.timeout),
        log_path=args.log,
    )
    result = RobotController(client, mode=args.mode).run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
