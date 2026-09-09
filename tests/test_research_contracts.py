"""Data-free tests for numerical, validation and reproducibility contracts."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import warnings

import numpy as np
import pandas as pd

from bsa_code.config import load_config
from bsa_code.demographics import _harmonise_age, _harmonise_education
from bsa_code.harmonisation_utils import clean_weight, rescale_attitude
from bsa_code.model_preprocessing import _build_preprocessor
from bsa_code.provenance import config_sha256, file_sha256, read_tuned_alpha
from bsa_code.recoding import weighted_quantile
from bsa_code.scoring_thresholds import _thresholds_for_rule
from bsa_code.train_eval import _primary_train_test_split, _regression_metrics, _split_holdout
from run_all import validate_output_paths

ROOT = Path(__file__).resolve().parents[1]


class NumericalContracts(unittest.TestCase):
    def test_reverse_coding_and_nonresponse(self):
        values = pd.Series([1, 3, 5, 8, 99, np.inf])
        actual = rescale_attitude(values, spec={"min": 1, "max": 5, "direction": "reverse", "exclude_codes": [8]}, missing_codes=[99])
        np.testing.assert_allclose(actual, [1, 0.5, 0, np.nan, np.nan, np.nan], equal_nan=True)

    def test_only_finite_positive_weights_survive(self):
        actual = clean_weight(pd.Series([1, 0, -1, np.inf, 99]), [99])
        np.testing.assert_allclose(actual, [1, np.nan, np.nan, np.nan, np.nan], equal_nan=True)

    def test_weighted_quantile_ignores_invalid_rows(self):
        self.assertEqual(weighted_quantile([0, 1, 100, np.nan], [1, 3, -1, 1], [0.5]), [1 / 3])
        self.assertTrue(np.isnan(weighted_quantile([1], [0], [0.8])[0]))

    def test_threshold_fits_reference_rows_only(self):
        reference = pd.DataFrame({"score": [0.1, 0.2, 0.3, 0.4], "weight": [1, 1, 1, 1]})
        pooled = pd.concat([reference, pd.DataFrame({"score": [1.0], "weight": [1000]})], ignore_index=True)
        expected = _thresholds_for_rule("top_20", reference, ["score"], "weight", True, 42)
        actual = _thresholds_for_rule("top_20", pooled, ["score"], "weight", True, 42,
                                      eligibility_masks={"score": pooled.index < 4})
        self.assertEqual(expected, actual)

    def test_weighted_metrics_against_hand_calculation(self):
        result = _regression_metrics([0, 1, 0], [0, 0, 1], [1, 2, 1])
        self.assertAlmostEqual(result["rmse"], np.sqrt(0.75))
        self.assertAlmostEqual(result["mae"], 0.75)
        self.assertAlmostEqual(result["r2"], -2)
        self.assertIsNone(_regression_metrics([1, 1], [0, 0])["r2"])


class ScientificBoundaries(unittest.TestCase):
    def test_holdout_is_disjoint_and_split_is_repeatable(self):
        frame = pd.DataFrame({"respondent_id": range(100), "wave_year": [2011] * 40 + [2024] * 40 + [2023] * 20,
                              "rwp_score": np.linspace(0, 1, 100), "weight": 1.0})
        settings = SimpleNamespace(holdout_years={2023}, train_size=0.7, project_seed=42, stratify_on_year=False)
        development, holdout = _split_holdout(frame, "rwp_score", settings)
        train, test, _ = _primary_train_test_split(development, settings, logging.getLogger("test"))
        train2, test2, _ = _primary_train_test_split(development, settings, logging.getLogger("test"))
        self.assertEqual(len(train), 56)
        self.assertEqual(len(test), 24)
        self.assertEqual(len(holdout), 20)
        self.assertTrue(set(train.index).isdisjoint(test.index))
        self.assertTrue(set(development.index).isdisjoint(holdout.index))
        pd.testing.assert_frame_equal(train, train2)
        pd.testing.assert_frame_equal(test, test2)

    def test_numeric_imputer_learns_from_training_only(self):
        processor = _build_preprocessor(["age"], ["education"], [["low", "high", "missing"]])
        processor.fit(pd.DataFrame({"age": [10, 20], "education": ["low", "high"]}))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            result = processor.transform(pd.DataFrame({"age": [np.nan, 1000], "education": ["new", "high"]}))
        if hasattr(result, "toarray"):
            result = result.toarray()
        self.assertEqual(result[0, 0], 15)
        self.assertEqual(processor.named_transformers_["num"]["imputer"].statistics_[0], 15)

    def test_education_correction_is_explicit_and_matches_earlier_higher_education(self):
        early = _harmonise_education(pd.Series([3]), year=2013, missing_codes=[])
        submitted = _harmonise_education(pd.Series([2, 3]), year=2024, missing_codes=[])
        corrected = _harmonise_education(pd.Series([2, 3]), year=2024, missing_codes=[], mapping_overrides={"2": "high"})
        self.assertEqual(submitted.tolist(), ["mid", "mid"])
        self.assertEqual(corrected.tolist(), [early.iloc[0], "mid"])

    def test_age_topcode_is_preserved_only_when_explicitly_configured(self):
        original = _harmonise_age(pd.Series([97, 98, 99]), year=2013, source="RAge", missing_codes=[97, 98, 99])
        corrected = _harmonise_age(pd.Series([97, 98, 99]), year=2013, source="RAge", missing_codes=[98, 99])
        self.assertTrue(original.isna().all())
        np.testing.assert_allclose(corrected, [97, np.nan, np.nan], equal_nan=True)


class ReproducibilityContracts(unittest.TestCase):
    def test_inherited_profiles_preserve_submitted_settings(self):
        original = load_config(ROOT / "config/code_config.json")
        reproduced = load_config(ROOT / "config/reproduce.json")
        corrected = load_config(ROOT / "config/corrected.json")
        self.assertEqual(original["waves"], reproduced["waves"])
        self.assertEqual(original["modelling"], reproduced["modelling"])
        self.assertNotIn("demographic_overrides", reproduced)
        self.assertIn("demographic_overrides", corrected)
        self.assertNotEqual(original["paths"]["outputs_dir"], reproduced["paths"]["outputs_dir"])
        validate_output_paths(reproduced)
        validate_output_paths(corrected)

    def test_config_extension_cycles_fail(self):
        with TemporaryDirectory() as folder:
            p = Path(folder) / "cycle.json"
            p.write_text('{"extends": "cycle.json"}')
            with self.assertRaisesRegex(ValueError, "cycle"):
                load_config(p)

    def test_output_protection_covers_ancestors_inputs_and_symlinks(self):
        for bad in [ROOT, ROOT / "outputs", ROOT / "outputs/main", ROOT / "raw_sav", Path("/")]:
            config = load_config(ROOT / "config/reproduce.json")
            config["paths"]["derived_dir"] = str(bad)
            with self.assertRaises(ValueError):
                validate_output_paths(config)
        config = load_config(ROOT / "config/reproduce.json")
        config["paths"]["outputs_dir"] = config["paths"]["derived_dir"] + "/nested"
        with self.assertRaises(ValueError):
            validate_output_paths(config)
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "outputs").mkdir()
            (root / "outputs" / "linked").symlink_to(root / "raw_sav")
            config["paths"]["derived_dir"] = str(root / "outputs" / "linked")
            with self.assertRaises(ValueError):
                validate_output_paths(config, root)

    def test_tuning_handoff_rejects_stale_config_or_data(self):
        with TemporaryDirectory() as folder:
            data = Path(folder) / "frame.bin"
            data.write_bytes(b"synthetic frame")
            decision_path = Path(folder) / "decision.json"
            config = {"seed": 42}
            self.assertIsNone(read_tuned_alpha(config, decision_path, data))
            decision_path.write_text(json.dumps({"input_fingerprint": {"configuration": config_sha256(config), "model_frame": file_sha256(data)},
                                                  "cv": {"selected": {"alpha": 30}}}))
            self.assertEqual(read_tuned_alpha(config, decision_path, data), 30)
            with self.assertRaisesRegex(ValueError, "stale"):
                read_tuned_alpha({"seed": 7}, decision_path, data)
            data.write_bytes(b"changed synthetic frame")
            with self.assertRaisesRegex(ValueError, "stale"):
                read_tuned_alpha(config, decision_path, data)


if __name__ == "__main__":
    unittest.main()
