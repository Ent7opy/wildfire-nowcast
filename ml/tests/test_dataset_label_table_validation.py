import unittest

from ml.denoiser.sql_safety import validate_table_reference


class TestLabelTableValidation(unittest.TestCase):
    def test_allows_simple_table_name(self):
        self.assertEqual(validate_table_reference("fire_labels"), "fire_labels")

    def test_allows_schema_qualified_table_name(self):
        self.assertEqual(validate_table_reference("public.fire_labels"), "public.fire_labels")

    def test_rejects_whitespace_and_sql_syntax(self):
        with self.assertRaises(ValueError):
            validate_table_reference("fire_labels l ON 1=1")

        with self.assertRaises(ValueError):
            validate_table_reference("fire_labels; DROP TABLE fire_detections;--")

    def test_rejects_quotes_and_special_chars(self):
        with self.assertRaises(ValueError):
            validate_table_reference('"fire_labels"')

        with self.assertRaises(ValueError):
            validate_table_reference("public.fire-labels")


if __name__ == "__main__":
    unittest.main()

