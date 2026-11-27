import unittest

import torch

from app_detection import selectionner_detections


class SelectionnerDetectionsTests(unittest.TestCase):
    def test_doublons_de_classes_differentes(self):
        boites = torch.tensor([
            [0., 0., 100., 100.],
            [2., 2., 102., 102.],
        ])
        scores = torch.tensor([0.75, 0.9])

        self.assertEqual(selectionner_detections(boites, scores, 0.5), [1])

    def test_avions_rapproches_et_seuil(self):
        boites = torch.tensor([
            [0., 0., 100., 100.],
            [45., 0., 145., 100.],
            [200., 0., 300., 100.],
        ])
        scores = torch.tensor([0.8, 0.9, 0.49])

        self.assertEqual(selectionner_detections(boites, scores, 0.5), [1, 0])

    def test_sans_detection(self):
        self.assertEqual(
            selectionner_detections(torch.empty((0, 4)), torch.empty(0), 0.5),
            [],
        )
        self.assertEqual(
            selectionner_detections(
                torch.tensor([[0., 0., 100., 100.]]), torch.tensor([0.49]), 0.5
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
