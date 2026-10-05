import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('cost_model',ROOT/'tools/cost_model.py')
cost=importlib.util.module_from_spec(spec); spec.loader.exec_module(cost)

class EvidenceTests(unittest.TestCase):
    def test_history_coverage(self):
        d=json.loads((ROOT/'history.json').read_text())
        self.assertEqual([x['id'] for x in d['track3']],list(range(1,47)))
        self.assertEqual(sum(x['world_record'] for x in d['track3']),20)
        self.assertEqual({x['id'] for x in d['track1']},set(range(1,93)))
        self.assertEqual(d['track3'][-1]['steps'],2690)
    def test_measured_pilots(self):
        d=json.loads((ROOT/'baselines.json').read_text())
        self.assertEqual(d['track3_record46']['steps'],2690)
        self.assertAlmostEqual(d['track1_record91']['training_seconds'],776.415)
        self.assertEqual(d['track1_record92']['status'],'completed_adapted_pilot')
        self.assertEqual(d['track1_record92']['steps'],1194)
        self.assertFalse(d['track1_record92']['certified_source_reproduction'])
        self.assertFalse(d['track1_record92']['statistical_parity'])
        self.assertAlmostEqual(d['track1_record92']['scored_loss'],3.280118829011917)
    def test_cost(self):
        r=cost.estimate(.95,.1,1.5)
        self.assertAlmostEqual(r['training_time_ratio'],.9975)
        self.assertAlmostEqual(cost.estimate(1,.1,2)['break_even_step_ratio'],1/1.1)
    def test_invalid_cost(self):
        for args in [(0,.1,1),(1,-.1,1),(1,1.1,1),(1,.1,-1),(float('nan'),.1,1),(1,1,0)]:
            with self.assertRaises(ValueError): cost.estimate(*args)

if __name__=='__main__': unittest.main()
