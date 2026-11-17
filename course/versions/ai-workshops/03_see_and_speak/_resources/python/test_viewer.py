"""Offline regression checks for native MJPEG/SVG and asynchronous detection.

These checks do not replace viewing the hardware stream on the Mac.
"""
from pathlib import Path
import contextlib
import io
import random
import builtins
import sys
from types import SimpleNamespace
import threading
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from main import Trigger, sentence, respond, main
from stream import Stream
from vision import DetectorWorker, Detection, load_labels, preprocess
from config import LABELS_PATH
from personality import Personality, RESPONSES, FALLBACK_RESPONSES


class RegressionTests(unittest.TestCase):
    def test_main_silent_humor_without_speech_imports(self):
        original_import = builtins.__import__
        def guard(name, *args, **kwargs):
            if name in {'speech', 'elevenlabs', 'openai', 'dotenv'}:
                raise AssertionError(f'Silent mode imported {name}')
            return original_import(name, *args, **kwargs)
        for humor in (False, True):
            with self.subTest(humor=humor):
                model = Mock()
                detections = [Detection('person', .99, (0,0,1,1)), Detection('banana', .9, (0,0,1,1))]
                model.detect.side_effect = [detections] * 3 + [KeyboardInterrupt()]
                output = io.StringIO()
                arguments = ['main.py', '--silent', '--only', 'banana'] + (['--humor'] if humor else [])
                with patch.object(sys, 'argv', arguments), patch.dict(sys.modules, {'vision': SimpleNamespace(Vision=Mock(return_value=model))}), patch('main.time.sleep'), patch('builtins.__import__', side_effect=guard), contextlib.redirect_stdout(output):
                    main()
                responses = [line.split(': ', 1)[1] for line in output.getvalue().splitlines() if line.startswith('NB3 (silent): ')]
                self.assertEqual(len(responses), 1)
                self.assertIn(responses[0], RESPONSES['banana'] if humor else ['I see a banana!'])
                model.close.assert_called_once()

    def test_humor_collection_matches_coco_and_stays_short(self):
        labels = load_labels(LABELS_PATH)
        for label, lines in RESPONSES.items():
            self.assertIn(label, labels)
            self.assertGreaterEqual(len(set(lines)), 3)
            for line in lines:
                self.assertLessEqual(len(line.split()), 25)
                self.assertLessEqual(sum(line.count(mark) for mark in '.!?'), 2)
                self.assertIn(label if label != 'person' else 'human' if 'human' in line.lower() else 'person', line.lower())

    def test_humor_variety_and_per_object_repetition(self):
        personality = Personality(random.Random(17))
        for label in list(RESPONSES) + ['toaster', 'scissors']:
            previous = None
            seen = set()
            expected = {line.format(object=label) for line in RESPONSES.get(label, FALLBACK_RESPONSES)}
            for _ in range(30):
                line = personality(label)
                self.assertIn(line, expected)
                self.assertNotEqual(line, previous)
                seen.add(line)
                previous = line
                personality('clock')  # Another class must not erase this one's memory.
            self.assertEqual(seen, expected)

    def test_silent_humor_filter_stability_cooldown(self):
        personality = Personality(random.Random(4))
        detections = [Detection('person', .99, (0, 0, 1, 1)), Detection('banana', .9, (0, 0, 1, 1))]
        trigger = Trigger(3, 8)
        output = io.StringIO()
        with patch('main.time.monotonic', side_effect=[0, .2, .4, .4, 1, 8.4, 8.6, 8.8, 8.8]), contextlib.redirect_stdout(output):
            respond(detections, trigger, None, True, 'banana', personality)
            respond(detections, trigger, None, True, 'banana', personality)
            self.assertEqual(personality.last_by_label, {})
            respond(detections, trigger, None, True, 'banana', personality)
            first = personality.last_by_label['banana']
            respond(detections, trigger, None, True, 'banana', personality)
            self.assertEqual(personality.last_by_label['banana'], first)
            for _ in range(3):
                respond(detections, trigger, None, True, 'banana', personality)
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertNotEqual(lines[0], lines[1])
        self.assertTrue(all('banana' in line.lower() for line in lines))
        self.assertNotIn('person', personality.last_by_label)

    def test_only_filter_stability_and_cooldown(self):
        person = Detection('person', .99, (0, 0, 1, 1))
        bottle = Detection('bottle', .87, (.1, .2, .8, .9))
        detections = [person, bottle]
        speaker, trigger = Mock(), Trigger(3, 8)
        # Higher confidence in a different class must not mask the target.
        with patch('main.time.monotonic', side_effect=[0, .2, .4, .4, 1, 8.4, 8.6, 8.8, 8.8]), contextlib.redirect_stdout(io.StringIO()):
            for _ in range(7):
                respond(detections, trigger, speaker, False, 'bottle')
        self.assertEqual(speaker.say.call_count, 2)
        speaker.say.assert_called_with('I see a bottle!')
        self.assertEqual(detections, [person, bottle])

    def test_only_missing_target_resets_stability(self):
        bottle = Detection('bottle', .9, (0, 0, 1, 1))
        banana = Detection('banana', .95, (0, 0, 1, 1))
        trigger, speaker = Trigger(3, 8), Mock()
        with contextlib.redirect_stdout(io.StringIO()):
            respond([bottle], trigger, speaker, False, 'bottle')
            respond([bottle], trigger, speaker, False, 'bottle')
            respond([banana], trigger, speaker, False, 'bottle')
            respond([bottle], trigger, speaker, False, 'bottle')
            respond([bottle], trigger, speaker, False, 'bottle')
            speaker.say.assert_not_called()
            respond([bottle], trigger, speaker, False, 'bottle')
        speaker.say.assert_called_once_with('I see a bottle!')

    def test_only_silent_and_default_selection(self):
        detections = [Detection('person', .99, (0, 0, 1, 1)), Detection('banana', .9, (0, 0, 1, 1))]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            respond(detections, Trigger(1, 8), None, True, 'banana')
            respond(detections, Trigger(1, 8), None, True)
        self.assertIn('NB3 (silent): I see a banana!', output.getvalue())
        self.assertIn('NB3 (silent): I see a person!', output.getvalue())

    def test_native_video(self):
        html = (Path(__file__).parent / "viewer.html").read_text()
        self.assertIn('src="/stream/camera.mjpg"', html)
        self.assertIn('object-fit: contain', html)
        self.assertIn('preserveAspectRatio="xMidYMid meet"', html)
        for removed in ('createImageBitmap', 'getContext(', 'ReadableStream', '<canvas'):
            self.assertNotIn(removed, html)
        self.assertIn('requestFullscreen()', html)

    def test_jpeg_passthrough(self):
        stream = Stream.__new__(Stream)
        stream.condition = threading.Condition()
        stream.sequence = 0
        jpeg = b"already encoded JPEG"
        sequence, timestamp = stream.publish(jpeg)
        self.assertIs(stream.jpeg, jpeg)
        self.assertEqual(sequence, 1)
        stream.set_detections([Detection('banana', .87, (.1, .2, .8, .9))], sequence, timestamp, 22)
        self.assertEqual(stream.result['detections'][0], {'label':'banana', 'score':.87, 'bbox':(.1,.2,.8,.9)})

    def test_preprocessing_and_trigger(self):
        image = np.full((600,800,3), (10,20,30), np.uint8)
        tensor = preprocess(image,[1,300,300,3],'uint8')
        self.assertEqual(tensor[0,0,0].tolist(), [30,20,10])
        self.assertEqual(load_labels(LABELS_PATH)[51], 'banana')
        trigger = Trigger(3, 8)
        self.assertFalse(trigger.update('banana',0))
        self.assertFalse(trigger.update('banana',.2))
        self.assertTrue(trigger.update('banana',.4))
        trigger.spoken(.4)
        self.assertFalse(trigger.update('banana',8.39))
        self.assertFalse(trigger.update('banana',8.4))
        self.assertFalse(trigger.update('banana',8.6))
        self.assertTrue(trigger.update('banana',8.8))
        self.assertEqual(sentence('banana'), 'I see a banana!')

    def test_latest_only_worker(self):
        entered, release, done = threading.Event(), threading.Event(), threading.Event()
        model = Mock()
        def detect(frame):
            entered.set()
            release.wait(2)
            return [Detection('banana',.9,(0,0,1,1))]
        model.detect.side_effect = detect
        results = []
        def result(detections, sequence, timestamp, milliseconds):
            results.append(sequence)
            if sequence == 100:
                done.set()
        success, encoded = cv2.imencode('.jpg',np.zeros((30,30,3),np.uint8))
        self.assertTrue(success)
        with patch('vision.Vision',return_value=model):
            worker = DetectorWorker(.5,result)
            try:
                worker.submit(encoded.tobytes(),1,time.monotonic())
                self.assertTrue(entered.wait(2))
                start = time.monotonic()
                for sequence in range(2,101):
                    worker.submit(encoded.tobytes(),sequence,time.monotonic())
                self.assertLess(time.monotonic()-start,.2)
                self.assertEqual(worker.pending[1],100)
                release.set()
                self.assertTrue(done.wait(3))
            finally:
                release.set()
                worker.close()
        self.assertEqual(results,[1,100])
        model.start_camera.assert_not_called()
        model.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
