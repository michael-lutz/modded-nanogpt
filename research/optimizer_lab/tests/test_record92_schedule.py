"""CPU-only audit of the GH200 full runner's schedule and cycle boundaries."""
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
# Loading the harness does not load FA3 or allocate CUDA memory until main().
spec = importlib.util.spec_from_file_location('record92_runner', ROOT/'research/optimizer_lab/runners/gh200_record92.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
import torch
from track_1_short.config import TRAINING_STAGES, LR_COOLDOWN_FRAC, SPLIT_EMBED_STAGE, WS_POST_YARN_EXT
from track_1_short.schedule import TrainingSchedule
from track_1_short.ngram_table import is_update_step


class FullScheduleTests(unittest.TestCase):
    def setUp(self):
        self.schedule = TrainingSchedule(TRAINING_STAGES,1122,20,torch.device('cpu'),
                                        LR_COOLDOWN_FRAC,SPLIT_EMBED_STAGE,WS_POST_YARN_EXT)

    def test_exact_schedule(self):
        self.assertEqual(self.schedule.total_steps,1194)
        self.assertEqual(self.schedule.boundaries,[(0,320),(320,681),(681,1107),(1107,1174),(1174,1194)])
        self.assertEqual(self.schedule.split_step,1175)
        keys = {runner.runner_key(self.schedule,s) for s in range(1194)}
        self.assertEqual(keys,{(0,10240),(1,10240),(2,14336),(2,24576),(3,0),(4,0)})

    def test_cycles_cover_steps_and_preserve_unflushed_tail(self):
        cycles = runner.cycles_for(1194)
        self.assertEqual([s for c in cycles for s in c],list(range(1194)))
        self.assertEqual(sum(is_update_step(s) for s in range(1194)),382)
        self.assertEqual(cycles[-1],[1192,1193])
        self.assertFalse(is_update_step(1193))
        self.assertTrue(all(is_update_step(c[-1]) for c in cycles[:-1]))
        self.assertIn([680,681,682,683],cycles)
        self.assertEqual(max(map(len,cycles)),4)

    def test_validation_preserves_rank_boundaries(self):
        # Five official global batches * eight ranks equals these forty contiguous slices.
        chunks=[(s*8+r)*262144 for s in range(5) for r in range(8)]
        self.assertEqual(chunks,list(range(0,10485760,262144)))


if __name__=='__main__':unittest.main()
