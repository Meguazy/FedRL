"""
Unit tests for eco_classifier.py

Tests the ECOClassifier class to ensure correct classification of chess games
into tactical or positional playstyles based on ECO opening codes.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
chess_fl_dir = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(chess_fl_dir))

from data.eco_classifier import ECOClassifier, classify_game_by_eco, PlaystyleType


class TestECOClassifier:
    """Test suite for ECOClassifier class."""

    @pytest.fixture
    def classifier(self):
        """Create an ECOClassifier instance for testing."""
        return ECOClassifier()

    # Initialization tests
    def test_classifier_initialization(self, classifier):
        """Test that classifier initializes correctly."""
        assert classifier is not None
        assert len(classifier.tactical_codes) > 0
        assert len(classifier.positional_codes) > 0

    def test_classifier_has_tactical_codes(self, classifier):
        """Test that classifier has tactical ECO codes defined."""
        assert "B90" in classifier.tactical_codes  # Sicilian Najdorf
        assert "B70" in classifier.tactical_codes  # Sicilian Dragon
        assert "C33" in classifier.tactical_codes  # King's Gambit Accepted

    def test_classifier_has_positional_codes(self, classifier):
        """Test that classifier has positional ECO codes defined."""
        assert "D63" in classifier.positional_codes  # QGD Orthodox
        assert "D10" in classifier.positional_codes  # Slav Defense
        assert "E20" in classifier.positional_codes  # Nimzo-Indian

    # Classification tests - Tactical openings
    def test_classify_sicilian_najdorf(self, classifier):
        """Test classification of Sicilian Najdorf (B90) as tactical."""
        result = classifier.classify("B90")
        assert result == "tactical"

    def test_classify_sicilian_dragon(self, classifier):
        """Test classification of Sicilian Dragon (B70) as tactical."""
        result = classifier.classify("B70")
        assert result == "tactical"

    def test_classify_kings_gambit(self, classifier):
        """Test classification of King's Gambit (C33) as tactical."""
        result = classifier.classify("C33")
        assert result == "tactical"

    def test_classify_sicilian_variants(self, classifier):
        """Test that various Sicilian variants are tactical."""
        sicilian_codes = ["B20", "B33", "B35", "B50", "B90", "B95"]
        for code in sicilian_codes:
            result = classifier.classify(code)
            assert result == "tactical", f"{code} should be tactical"

    # Classification tests - Positional openings
    def test_classify_queens_gambit_declined(self, classifier):
        """Test classification of QGD (D63) as positional."""
        result = classifier.classify("D63")
        assert result == "positional"

    def test_classify_slav_defense(self, classifier):
        """Test classification of Slav Defense (D10) as positional."""
        result = classifier.classify("D10")
        assert result == "positional"

    def test_classify_nimzo_indian(self, classifier):
        """Test classification of Nimzo-Indian (E20) as positional."""
        result = classifier.classify("E20")
        assert result == "positional"

    def test_classify_caro_kann(self, classifier):
        """Test classification of Caro-Kann (B12) as positional."""
        result = classifier.classify("B12")
        assert result == "positional"

    def test_classify_queens_gambit_variants(self, classifier):
        """Test that various QGD variants are positional."""
        qgd_codes = ["D30", "D37", "D45", "D53", "D63"]
        for code in qgd_codes:
            result = classifier.classify(code)
            assert result == "positional", f"{code} should be positional"

    # Edge cases and input validation
    def test_classify_empty_string(self, classifier):
        """Test classification with empty string returns default."""
        result = classifier.classify("")
        assert result == "positional"  # Default

    def test_classify_none(self, classifier):
        """Test classification with None returns default."""
        result = classifier.classify(None)
        assert result == "positional"  # Default

    def test_classify_invalid_eco_code(self, classifier):
        """Test classification with invalid ECO code returns default."""
        result = classifier.classify("Z99")
        assert result == "positional"  # Default

    def test_classify_with_suffix(self, classifier):
        """Test classification handles ECO codes with suffixes (e.g., B90a)."""
        # ECO codes sometimes have letter suffixes
        result = classifier.classify("B90a")
        assert result == "tactical"  # Should still recognize B90

    def test_classify_lowercase(self, classifier):
        """Test classification handles lowercase ECO codes."""
        result = classifier.classify("b90")
        assert result == "tactical"  # Should normalize to uppercase

    def test_classify_with_custom_default(self, classifier):
        """Test classification with custom default value."""
        result = classifier.classify("Z99", default="tactical")
        assert result == "tactical"

    # Helper method tests
    def test_is_tactical(self, classifier):
        """Test is_tactical helper method."""
        assert classifier.is_tactical("B90") is True
        assert classifier.is_tactical("D63") is False

    def test_is_positional(self, classifier):
        """Test is_positional helper method."""
        assert classifier.is_positional("D63") is True
        assert classifier.is_positional("B90") is False

    # Opening examples tests
    def test_get_opening_examples_tactical(self, classifier):
        """Test getting tactical opening examples."""
        examples = classifier.get_opening_examples("tactical")
        assert len(examples) > 0
        assert isinstance(examples, list)
        assert "Sicilian" in " ".join(examples)

    def test_get_opening_examples_positional(self, classifier):
        """Test getting positional opening examples."""
        examples = classifier.get_opening_examples("positional")
        assert len(examples) > 0
        assert isinstance(examples, list)
        assert "Gambit" in " ".join(examples) or "Queen" in " ".join(examples)

    def test_get_opening_examples_invalid_playstyle(self, classifier):
        """Test getting examples with invalid playstyle raises error."""
        with pytest.raises(ValueError):
            classifier.get_opening_examples("invalid")

    # Statistics tests
    def test_get_statistics(self, classifier):
        """Test getting classifier statistics."""
        stats = classifier.get_statistics()
        assert "total_eco_codes" in stats
        assert "tactical_codes" in stats
        assert "positional_codes" in stats
        assert stats["total_eco_codes"] > 0
        assert stats["tactical_codes"] > 0
        assert stats["positional_codes"] > 0

    def test_statistics_sum_correct(self, classifier):
        """Test that tactical + positional equals total."""
        stats = classifier.get_statistics()
        assert stats["tactical_codes"] + stats["positional_codes"] == stats["total_eco_codes"]

    # Consistency tests
    def test_no_overlap_between_tactical_and_positional(self, classifier):
        """Test that no ECO code is in both tactical and positional sets."""
        overlap = classifier.tactical_codes & classifier.positional_codes
        assert len(overlap) == 0, f"Found overlap: {overlap}"

    def test_all_codes_are_three_characters(self, classifier):
        """Test that all ECO codes are 3 characters (e.g., B90, D30)."""
        all_codes = classifier.tactical_codes | classifier.positional_codes
        for code in all_codes:
            assert len(code) == 3, f"Code {code} is not 3 characters"
            assert code[0].isalpha(), f"Code {code} doesn't start with letter"
            assert code[1:].isdigit(), f"Code {code} doesn't end with digits"

    # Specific opening coverage tests
    def test_covers_major_tactical_openings(self, classifier):
        """Test that major tactical openings are covered."""
        major_tactical = {
            "B20": "Sicilian Defense",
            "C30": "King's Gambit",
            "C50": "Italian Game",
            "B03": "Alekhine's Defense",
            "C21": "Center Game",
        }
        for code, name in major_tactical.items():
            result = classifier.classify(code)
            assert result == "tactical", f"{name} ({code}) should be tactical"

    def test_covers_major_positional_openings(self, classifier):
        """Test that major positional openings are covered."""
        major_positional = {
            "D30": "Queen's Gambit Declined",
            "D10": "Slav Defense",
            "E20": "Nimzo-Indian",
            "B12": "Caro-Kann",
            "A30": "English Opening",
        }
        for code, name in major_positional.items():
            result = classifier.classify(code)
            assert result == "positional", f"{name} ({code}) should be positional"

    # Consistency with templates (from your opening database)
    def test_matches_template_tactical_openings(self, classifier):
        """Test that classifier matches the tactical openings from templates."""
        template_tactical_codes = ["B70", "B90", "B35", "C33", "C30", "C51", "C50", "B03", "C25", "C21", "B01"]
        for code in template_tactical_codes:
            result = classifier.classify(code)
            assert result == "tactical", f"Template tactical opening {code} not classified correctly"

    def test_matches_template_positional_openings(self, classifier):
        """Test that classifier matches the positional openings from templates."""
        template_positional_codes = ["D63", "D34", "D10", "E20", "E12", "E00", "A30", "A25", "A09", "E11"]
        for code in template_positional_codes:
            result = classifier.classify(code)
            assert result == "positional", f"Template positional opening {code} not classified correctly"


class TestClassifyGameByEco:
    """Test suite for the convenience function classify_game_by_eco."""

    def test_convenience_function_tactical(self):
        """Test convenience function with tactical opening."""
        result = classify_game_by_eco("B90")
        assert result == "tactical"

    def test_convenience_function_positional(self):
        """Test convenience function with positional opening."""
        result = classify_game_by_eco("D63")
        assert result == "positional"

    def test_convenience_function_creates_new_classifier(self):
        """Test that convenience function works independently."""
        result1 = classify_game_by_eco("B90")
        result2 = classify_game_by_eco("D63")
        assert result1 == "tactical"
        assert result2 == "positional"


class TestPlaystyleType:
    """Test suite for PlaystyleType type hint."""

    def test_playstyle_type_exists(self):
        """Test that PlaystyleType is defined."""
        assert PlaystyleType is not None

    def test_playstyle_type_values(self):
        """Test that PlaystyleType has correct literal values."""
        # This is a type hint test - mainly for documentation
        # The actual runtime check would be in type checkers like mypy
        tactical: PlaystyleType = "tactical"
        positional: PlaystyleType = "positional"
        assert tactical == "tactical"
        assert positional == "positional"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
