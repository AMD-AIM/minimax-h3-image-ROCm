"""Static DiT placement regression checks; no GPU is required."""
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from minimax_h3_image_rocm.weights import SafetensorIndex, plan_dit
except ImportError as error:  # torch/safetensors live in the foundation image
    raise unittest.SkipTest(f"weights module unavailable: {error}")

TRANSFORMER = ROOT / "models/native/minimax-h3-image/transformer"


def fake_index(block_bytes, front=7, final=11, raw_head=13):
    specs = {f"blocks.{i}.weight": SimpleNamespace(bytes=value) for i, value in enumerate(block_bytes)}
    specs["video_patch_proj.weight"] = SimpleNamespace(bytes=front)
    specs["final_layer.adaln_proj.weight"] = SimpleNamespace(bytes=final)
    specs["final_layer.video_out.weight"] = SimpleNamespace(bytes=raw_head)
    return SimpleNamespace(specs=specs)


def released_four_way(block_bytes, front, final):
    """The released nested-loop search, kept as the four-device oracle."""
    prefix = [0]
    for value in block_bytes:
        prefix.append(prefix[-1] + value)
    best = None
    for first in range(1, 48):
        for second in range(first + 1, 49):
            for third in range(second + 1, 50):
                loads = [
                    front + prefix[first],
                    prefix[second] - prefix[first],
                    prefix[third] - prefix[second],
                    prefix[50] - prefix[third] + final,
                ]
                score = (max(loads), max(loads) - min(loads), sum(abs(loads[i] - loads[i - 1]) for i in range(1, 4)))
                if best is None or score < best[0]:
                    best = (score, (first, second, third), loads)
    return best[1], best[2]


class PlanTests(unittest.TestCase):
    def test_four_way_matches_released_search(self):
        generator = random.Random(20261114)
        cases = [[100] * 50, [100 + (i % 3) for i in range(50)]]
        cases += [[generator.randint(50, 150) for _ in range(50)] for _ in range(3)]
        for blocks in cases:
            plan = plan_dit(fake_index(blocks), 4, pdd_bytes=5)
            cuts, loads = released_four_way(blocks, 7, 11 + 5)
            self.assertEqual(tuple(row["start"] for row in plan["ranges"][1:]), cuts)
            self.assertEqual([row["bytes"] for row in plan["ranges"]], loads)

    def test_two_way_is_contiguous_and_balanced(self):
        blocks = [100 + (i % 5) for i in range(50)]
        plan = plan_dit(fake_index(blocks), 2, pdd_bytes=5)
        ranges = plan["ranges"]
        self.assertEqual([(r["start"], r["stop"]) for r in ranges][0][0], 0)
        self.assertEqual(ranges[0]["stop"], ranges[1]["start"])
        self.assertEqual(ranges[1]["stop"], 50)
        best = min(
            max(7 + sum(blocks[:cut]), sum(blocks[cut:]) + 16) for cut in range(1, 50)
        )
        self.assertEqual(plan["maximum_device_weight_bytes"], best)

    def test_unsupported_device_count(self):
        with self.assertRaises(ValueError):
            plan_dit(fake_index([1] * 50), 3)

    @unittest.skipUnless((TRANSFORMER / "model.safetensors.index.json").is_file(), "weights not downloaded")
    def test_released_weights_keep_published_four_way_ranges(self):
        plan = plan_dit(SafetensorIndex(TRANSFORMER), 4)
        self.assertEqual(
            [(row["start"], row["stop"]) for row in plan["ranges"]],
            [(0, 11), (11, 24), (24, 37), (37, 50)],
        )


if __name__ == "__main__":
    unittest.main()
