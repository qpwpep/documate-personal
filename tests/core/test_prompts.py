import unittest

from src.core.prompts import needs_search


class PromptsTest(unittest.TestCase):
    def test_needs_search_matches_library_explainer_request(self) -> None:
        self.assertTrue(needs_search("pandas에 대해 알려줘"))

    def test_needs_search_matches_korean_technical_request(self) -> None:
        self.assertTrue(needs_search("판다스의 성능 최적화를 알려줘"))


if __name__ == "__main__":
    unittest.main()
