import unittest
from funding_cleanup import clean_funding_name


class FundingCleanupTests(unittest.TestCase):
    def test_real_grants_preserved(self):
        for name in ['NSF grant 123456', 'Google Research', '国家自然科学基金', '[RGPIN-2020-04597, CFI-JELF Project 43994]']:
            self.assertEqual(clean_funding_name(name)[0], [name])

    def test_structured_list_unwrapped(self):
        self.assertEqual(clean_funding_name("{'funding': ['NSF 987654', 'Google Research', 'NSF 987654']}")[0], ['NSF 987654', 'Google Research'])

    def test_contaminated_examples_quarantined(self):
        for name in ['NSF grant 12345', "{'funding': ['NSF grant 12345', 'Google Research']}", "['.', 'End with }']"]:
            self.assertEqual(clean_funding_name(name)[0], [])

    def test_unknown_structure_not_guessed(self):
        name = "{'award': 'Potential real grant'}"
        self.assertEqual(clean_funding_name(name)[0], [name])


if __name__ == '__main__':
    unittest.main()
