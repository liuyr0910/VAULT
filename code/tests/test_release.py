"""CPU checks for portable data and evaluation boundary cases."""
import csv
import dataclasses
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'inf_script'))

def module(name, path):
    spec=importlib.util.spec_from_file_location(name, ROOT/path)
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj;spec.loader.exec_module(obj)
    return obj

class ReleaseTests(unittest.TestCase):
    @unittest.skipUnless((ROOT/'data/annotations/train.jsonl').is_file(), 'Separate dataset required')
    def test_dataset_and_training_contract(self):
        builder=module('build_sft', 'scripts/build_sft.py')
        rows=[json.loads(s) for s in (ROOT/'data/annotations/train.jsonl').read_text().splitlines()]
        records=builder.build_records(rows)
        self.assertEqual(len(records),6690)
        prepared=ROOT/'sft/data/all_tasks_mixed.json'
        if prepared.exists():self.assertEqual(records,json.loads(prepared.read_text()))
        self.assertEqual(len({r['id'] for r in records}),len(records))
        for row in records:
            self.assertTrue((ROOT/row['video']).is_file())
            self.assertIn(f"<video>{row['video']}</video>",row['conversations'][0]['value'])
        bad=dict(rows[0], video_path='../outside.mp4')
        with self.assertRaises(ValueError):builder.build_records([bad])

    def test_temporal_normal_and_overlap(self):
        temporal=module('temporal', 'eval_script/eval_tmp.py')
        self.assertEqual(temporal.parse_new_format('-1 - -1'),[[-1.,-1.]])
        self.assertEqual(temporal.calculate_global_iou([[-1,-1]],temporal.parse_new_format('-1 - -1')),1)
        self.assertEqual(temporal.calculate_global_iou([],[[0,2]]),0)
        self.assertAlmostEqual(temporal.calculate_global_iou([[0,3],[2,4]],[[2,6]]),1/3)

    @unittest.skipUnless((ROOT/'data/annotations/test.jsonl').is_file(), 'Separate dataset required')
    def test_classification_covers_every_label(self):
        classifier=module('classifier', 'eval_script/eval_cls.py')
        labels=set()
        for split in ('train','val','test'):
            for line in (ROOT/'data/annotations'/f'{split}.jsonl').read_text().splitlines():
                labels.update(x.lower() for x in json.loads(line)['category'])
        self.assertEqual(labels,set(classifier.CATEGORY_LIST))
        self.assertEqual(classifier.parse_categories('drowning, animal aggression',True),{'drowning','animal aggression'})

    def test_incomplete_circular_question_is_not_robust(self):
        evaluator=module('vqa', 'eval_script/eval_vqa-circular.py')
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'vqa.jsonl';out=Path(tmp)/'metrics.csv'
            rows=[dict(question_id='complete',gt_correct_key=c,prediction=c,shift_id=i) for i,c in enumerate('ABCD')]
            rows.append(dict(question_id='partial',gt_correct_key='A',prediction='A',shift_id=0))
            src.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            evaluator.evaluate_vqa_circular(src,out)
            with out.open() as handle:report=list(csv.DictReader(handle))
            indexed={x['question_id']:x for x in report}
            self.assertEqual(indexed['partial']['is_robust_correct'],'0')
            self.assertEqual(indexed['complete']['is_robust_correct'],'1')
            self.assertEqual(indexed['ROBUST ACCURACY']['is_robust_correct'],'50.00%')

    def test_event_cache_rejects_changed_configuration(self):
        from event_rgb_visualization import EventImageConfig,POLICY,validate_cached_event_meta
        c=EventImageConfig();meta=dict(policy=POLICY,config=dataclasses.asdict(c),config_signature=c.signature())
        validate_cached_event_meta(meta)
        meta['config']['sensor_width']=320
        with self.assertRaises(ValueError):validate_cached_event_meta(meta)

if __name__=='__main__':unittest.main()
