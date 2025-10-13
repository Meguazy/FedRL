"""
ECO (Encyclopedia of Chess Openings) classifier for chess games.

This module classifies chess games into tactical or positional clusters
based on their ECO opening code, matching the opening templates used
in the federated learning system.
"""

from typing import Literal, Tuple, List


# Tactical ECO codes based on opening templates
TACTICAL_ECO_CODES = {
    # Sicilian Defence variations
    "B70", "B90", "B35",  # Dragon, Najdorf, Accelerated Dragon
    "B20", "B21", "B22", "B23", "B24", "B25", "B26", "B27", "B28", "B29",  # Other Sicilians
    "B30", "B31", "B32", "B33", "B34", "B36", "B37", "B38", "B39",
    "B40", "B41", "B42", "B43", "B44", "B45", "B46", "B47", "B48", "B49",
    "B50", "B51", "B52", "B53", "B54", "B55", "B56", "B57", "B58", "B59",
    "B60", "B61", "B62", "B63", "B64", "B65", "B66", "B67", "B68", "B69",
    "B71", "B72", "B73", "B74", "B75", "B76", "B77", "B78", "B79",
    "B80", "B81", "B82", "B83", "B84", "B85", "B86", "B87", "B88", "B89",
    "B91", "B92", "B93", "B94", "B95", "B96", "B97", "B98", "B99",

    # King's Gambit
    "C30", "C33", "C34", "C35", "C36", "C37", "C38", "C39",

    # Italian Game aggressive lines
    "C50", "C51", "C52", "C53", "C54",

    # Alekhine's Defence
    "B02", "B03", "B04", "B05",

    # Vienna Game
    "C25", "C26", "C27", "C28", "C29",

    # Centre Game
    "C21", "C22",

    # Scandinavian Defence
    "B01",

    # Pirc/Modern Defence (sharp lines)
    "B07", "B08", "B09",

    # Ruy Lopez sharp lines (Marshall, Schliemann)
    "C80", "C81", "C82", "C83", "C84", "C85", "C86", "C87", "C88", "C89",
    "C90", "C91", "C92", "C93", "C94", "C95", "C96", "C97", "C98", "C99",

    # King's Indian Attack lines
    "E60", "E61", "E62", "E63", "E64", "E65", "E66", "E67", "E68", "E69",
    "E70", "E71", "E72", "E73", "E74", "E75", "E76", "E77", "E78", "E79",
    "E80", "E81", "E82", "E83", "E84", "E85", "E86", "E87", "E88", "E89",
    "E90", "E91", "E92", "E93", "E94", "E95", "E96", "E97", "E98", "E99",
}

# Positional ECO codes based on opening templates
POSITIONAL_ECO_CODES = {
    # Queen's Gambit Declined
    "D30", "D31", "D32", "D33", "D34", "D35", "D36", "D37", "D38", "D39",
    "D40", "D41", "D42", "D43", "D44", "D45", "D46", "D47", "D48", "D49",
    "D50", "D51", "D52", "D53", "D54", "D55", "D56", "D57", "D58", "D59",
    "D60", "D61", "D62", "D63", "D64", "D65", "D66", "D67", "D68", "D69",

    # Slav Defence
    "D10", "D11", "D12", "D13", "D14", "D15", "D16", "D17", "D18", "D19",

    # Semi-Slav
    "D43", "D44", "D45", "D46", "D47",

    # London System
    "D00", "D01", "D02", "D03", "D04", "D05",

    # Caro-Kann Defence
    "B10", "B11", "B12", "B13", "B14", "B15", "B16", "B17", "B18", "B19",

    # Nimzo-Indian Defence
    "E20", "E21", "E22", "E23", "E24", "E25", "E26", "E27", "E28", "E29",
    "E30", "E31", "E32", "E33", "E34", "E35", "E36", "E37", "E38", "E39",
    "E40", "E41", "E42", "E43", "E44", "E45", "E46", "E47", "E48", "E49",
    "E50", "E51", "E52", "E53", "E54", "E55", "E56", "E57", "E58", "E59",

    # Queen's Indian Defence
    "E12", "E13", "E14", "E15", "E16", "E17", "E18", "E19",

    # Bogo-Indian Defence
    "E11",

    # Catalan Opening
    "E00", "E01", "E02", "E03", "E04", "E05", "E06", "E07", "E08", "E09",

    # English Opening
    "A10", "A11", "A12", "A13", "A14", "A15", "A16", "A17", "A18", "A19",
    "A20", "A21", "A22", "A23", "A24", "A25", "A26", "A27", "A28", "A29",
    "A30", "A31", "A32", "A33", "A34", "A35", "A36", "A37", "A38", "A39",

    # Réti Opening
    "A04", "A05", "A06", "A07", "A08", "A09",

    # Ruy Lopez Berlin/positional lines
    "C65", "C66", "C67", "C68", "C69",

    # Grunfeld Defence
    "D70", "D71", "D72", "D73", "D74", "D75", "D76", "D77", "D78", "D79",
    "D80", "D81", "D82", "D83", "D84", "D85", "D86", "D87", "D88", "D89",
    "D90", "D91", "D92", "D93", "D94", "D95", "D96", "D97", "D98", "D99",
}


PlaystyleType = Literal["tactical", "positional"]


class ECOClassifier:
    """
    Classifier for chess games based on ECO opening codes.

    This class maps ECO codes to playstyles (tactical vs positional)
    to enable cluster-specific game filtering in federated learning.

    Example:
        >>> classifier = ECOClassifier()
        >>> classifier.classify("B90")  # Sicilian Najdorf
        'tactical'
        >>> classifier.classify("D63")  # Queen's Gambit Declined
        'positional'
    """

    def __init__(self):
        """Initialize the ECO classifier."""
        self.tactical_codes = TACTICAL_ECO_CODES
        self.positional_codes = POSITIONAL_ECO_CODES

    def classify(self, eco_code: str, default: PlaystyleType = "positional") -> PlaystyleType:
        """
        Classify a game by its ECO code.

        Args:
            eco_code: ECO code string like "B90", "D63", "C50"
            default: Default playstyle if ECO code doesn't match

        Returns:
            'tactical' or 'positional'

        Example:
            >>> classifier = ECOClassifier()
            >>> classifier.classify("B33")  # Sicilian Sveshnikov
            'tactical'
            >>> classifier.classify("D37")  # QGD
            'positional'
        """
        if not eco_code:
            return default

        # Normalize ECO code (remove suffixes like 'a', 'b')
        eco_normalized = eco_code[:3].upper()

        # Check exact match first
        if eco_normalized in self.tactical_codes:
            return "tactical"
        if eco_normalized in self.positional_codes:
            return "positional"

        return default

    def is_tactical(self, eco_code: str) -> bool:
        """
        Check if an ECO code represents a tactical opening.

        Args:
            eco_code: ECO code string

        Returns:
            True if tactical, False otherwise
        """
        return self.classify(eco_code) == "tactical"

    def is_positional(self, eco_code: str) -> bool:
        """
        Check if an ECO code represents a positional opening.

        Args:
            eco_code: ECO code string

        Returns:
            True if positional, False otherwise
        """
        return self.classify(eco_code) == "positional"

    def get_opening_examples(self, playstyle: PlaystyleType) -> List[str]:
        """
        Get example opening names for a playstyle.

        Args:
            playstyle: 'tactical' or 'positional'

        Returns:
            List of opening names with ECO codes
        """
        if playstyle == "tactical":
            return [
                "Sicilian Dragon (B70)",
                "Sicilian Najdorf (B90)",
                "Sicilian Accelerated Dragon (B35)",
                "King's Gambit Accepted (C33)",
                "King's Gambit Declined (C30)",
                "Evans Gambit (C51)",
                "Italian Game Aggressive (C50)",
                "Alekhine Defence Four Pawns (B03)",
                "Vienna Gambit (C25)",
                "Centre Game (C21)",
                "Scandinavian Main Line (B01)",
            ]
        elif playstyle == "positional":
            return [
                "Queen's Gambit Declined Orthodox (D63)",
                "Queen's Gambit Declined Tarrasch (D34)",
                "Slav Defence (D10)",
                "Nimzo-Indian Defence (E20)",
                "Queen's Indian Defence (E12)",
                "Catalan Opening (E00)",
                "English Opening Symmetrical (A30)",
                "English Opening Closed System (A25)",
                "Réti Opening (A09)",
                "Bogo-Indian Defence (E11)",
            ]
        else:
            raise ValueError(f"Invalid playstyle: {playstyle}")

    def get_statistics(self) -> dict:
        """
        Get statistics about the classifier.

        Returns:
            Dictionary with classifier statistics
        """
        return {
            "total_eco_codes": len(self.tactical_codes) + len(self.positional_codes),
            "tactical_codes": len(self.tactical_codes),
            "positional_codes": len(self.positional_codes),
        }


# Convenience function for quick classification
def classify_game_by_eco(eco_code: str) -> PlaystyleType:
    """
    Classify a game by ECO code (convenience function).

    Args:
        eco_code: ECO code like "B90", "D63"

    Returns:
        'tactical' or 'positional'

    Example:
        >>> classify_game_by_eco("B33")
        'tactical'
        >>> classify_game_by_eco("D37")
        'positional'
    """
    classifier = ECOClassifier()
    return classifier.classify(eco_code)


# if __name__ == "__main__":
#     # Test the classifier
#     classifier = ECOClassifier()

#     print("ECO Classifier Statistics")
#     print("=" * 50)
#     stats = classifier.get_statistics()
#     for key, value in stats.items():
#         print(f"{key}: {value}")

#     print("\n\nTesting Tactical Openings:")
#     print("=" * 50)
#     tactical_codes = ["B70", "B90", "B35", "C33", "C30", "C51", "C50", "B03", "C25", "C21", "B01"]
#     for code in tactical_codes:
#         result = classifier.classify(code)
#         print(f"  {code}: {result}")

#     print("\n\nTesting Positional Openings:")
#     print("=" * 50)
#     positional_codes = ["D63", "D34", "D10", "E20", "E12", "E00", "A30", "A25", "A09", "E11"]
#     for code in positional_codes:
#         result = classifier.classify(code)
#         print(f"  {code}: {result}")

#     print("\n\nTactical Opening Examples:")
#     print("=" * 50)
#     for opening in classifier.get_opening_examples("tactical"):
#         print(f"  - {opening}")

#     print("\n\nPositional Opening Examples:")
#     print("=" * 50)
#     for opening in classifier.get_opening_examples("positional"):
#         print(f"  - {opening}")
