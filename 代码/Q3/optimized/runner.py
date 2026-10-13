from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from optimized.controller import AnytimeRobotController
from common.communication import RetryPolicy, SimulatorClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CUMCM B Q3 anytime robot-dog solver")
    parser.add_argument("--robot-id", default=os.getenv("ROBOT_ID"))
    parser.add_argument("--base-url", default=os.getenv("SIMULATOR_URL", "http://127.0.0.1:2026"))
    parser.add_argument("--log", type=Path, default=Path("logs/robot_dog_anytime.jsonl"))
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--opportunity-cap", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.robot_id:
        args.robot_id = input("请输入当前模拟器登录的参赛队号：").strip()
    if not args.robot_id:
        raise SystemExit("参赛队号不能为空")
    if not 0 <= args.opportunity_cap <= 8:
        raise SystemExit("--opportunity-cap 必须在 0..8 之间")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    client = SimulatorClient(
        robot_id=args.robot_id,
        base_url=args.base_url,
        retry=RetryPolicy(attempts=args.retries, timeout_s=args.timeout),
        log_path=args.log,
    )
    result = AnytimeRobotController(
        client, max_opportunistic_per_stop=args.opportunity_cap
    ).run()
    result["algorithm"] = "q3_anytime_two_observation_roles"
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
