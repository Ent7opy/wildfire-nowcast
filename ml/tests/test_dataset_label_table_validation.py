import unittest

from ml.denoiser.sql_safety import validate_table_reference


class TestLabelTableValidation(unittest.TestCase):
    def test_allows_simple_table_name(self):
        self.assertEqual(validate_table_reference("denoiser_labels_v2"), "denoiser_labels_v2")

    def test_allows_schema_qualified_table_name(self):
        self.assertEqual(validate_table_reference("public.denoiser_labels_v2"), "public.denoiser_labels_v2")

    def test_rejects_whitespace_and_sql_syntax(self):
        with self.assertRaises(ValueError):
            validate_table_reference("denoiser_labels_v2 l ON 1=1")

        with self.assertRaises(ValueError):
            validate_table_reference("denoiser_labels_v2; DROP TABLE fire_detections;--")

    def test_rejects_quotes_and_special_chars(self):
        with self.assertRaises(ValueError):
            validate_table_reference('"fire_labels"')

        with self.assertRaises(ValueError):
            validate_table_reference("public.fire-labels")


if __name__ == "__main__":
    unittest.main()

