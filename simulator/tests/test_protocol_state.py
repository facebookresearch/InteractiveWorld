# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import unittest

from simulator.interaction.episode_loop import build_protocol, reset_protocol


def _protocol_args() -> argparse.Namespace:
    return argparse.Namespace(
        assistant_tokens=256,
        frame_max_side=512,
        human_url="http://127.0.0.1:8802/generate",
        max_frames=32,
        max_ticks=None,
        per_interval=4,
    )


class ProtocolStateTest(unittest.TestCase):
    def test_build_protocol_initializes_human_policy_state(self) -> None:
        protocol = build_protocol(_protocol_args())

        self.assertEqual(
            {
                "status": "LEGACY",
                "code": "",
                "path": "",
                "message": "",
            },
            protocol._human_policy_state,
        )

    def test_reset_protocol_reinitializes_human_policy_state(self) -> None:
        protocol = build_protocol(_protocol_args())
        protocol._human_policy_state.update(
            {
                "status": "COMPLETE",
                "code": "old-code",
                "path": "old-path",
                "message": "old-message",
            }
        )

        reset_protocol(protocol)

        self.assertEqual(
            {
                "status": "LEGACY",
                "code": "",
                "path": "",
                "message": "",
            },
            protocol._human_policy_state,
        )
