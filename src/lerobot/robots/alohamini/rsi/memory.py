#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""M1 of EmbodiedRSI's Hierarchical Memory (Sec. 3.3, Eq. 8): one record per physical episode.

M2 is the Hypothesis Graph; M3 (reusable cognition) is a list of rules kept by the evolution loop. The
trajectory itself is the LeRobot dataset episode a record points to, so only the harness-level view of the
episode is stored here.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ExperienceRecord:
    episode: int
    variant: list[str]
    baseline_version: int
    init_state: str
    repeat: int
    steps: list[dict]
    success: bool
    duration_s: float
    note: str = ""
    frames: list[str] = field(default_factory=list)
    timestamp: float = 0.0


class ExperienceStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def append(self, record: ExperienceRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as file:
            file.write(json.dumps(asdict(record)) + "\n")

    def load(self) -> list[ExperienceRecord]:
        if not self.path.exists():
            return []
        return [ExperienceRecord(**json.loads(line)) for line in self.path.read_text().splitlines() if line]
