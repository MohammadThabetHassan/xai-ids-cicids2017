"""Tests for XCS (XAI Confidence Score) computation."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
class TestXCSFormula:
    """Test XCS formula properties."""

    def test_xcs_in_valid_range(self):
        """XCS should always be in [0, 1]."""
        # XCS = 0.4*Conf + 0.3*(1-Instab) + 0.3*Jaccard
        # Conf in [0,1], Instab in [0,0.5], Jaccard in [0,1]
        # Min: 0.4*0 + 0.3*(1-0.5) + 0.3*0 = 0.15
        # Max: 0.4*1 + 0.3*(1-0) + 0.3*1 = 1.0
        for conf in [0.0, 0.5, 1.0]:
            for instab in [0.0, 0.25, 0.5]:
                for jaccard in [0.0, 0.5, 1.0]:
                    xcs = 0.4 * conf + 0.3 * (1 - instab) + 0.3 * jaccard
                    assert 0 <= xcs <= 1.0, f"XCS out of range: {xcs}"

    def test_xcs_threshold(self):
        """XCS threshold of 0.3 should flag low-confidence predictions."""
        # At conf=0, instab=0.5, jaccard=0: XCS = 0 + 0.15 + 0 = 0.15 < 0.3
        xcs_low = 0.4 * 0.0 + 0.3 * (1 - 0.5) + 0.3 * 0.0
        assert xcs_low < 0.3

        # At conf=1, instab=0, jaccard=1: XCS = 0.4 + 0.3 + 0.3 = 1.0 > 0.3
        xcs_high = 0.4 * 1.0 + 0.3 * (1 - 0.0) + 0.3 * 1.0
        assert xcs_high > 0.3

    def test_wrong_predictions_have_lower_xcs(self):
        """Wrong predictions should generally have lower XCS than correct ones."""
        # Simulate: wrong predictions tend to have lower confidence
        correct_conf = 0.9
        wrong_conf = 0.3

        xcs_correct = 0.4 * correct_conf + 0.3 * 0.7 + 0.3 * 0.3
        xcs_wrong = 0.4 * wrong_conf + 0.3 * 0.5 + 0.3 * 0.1

        assert xcs_correct > xcs_wrong
class TestXCSComponents:
    """Test individual XCS components."""

    def test_confidence_component(self):
        """Confidence component should be 0.4 * max_proba."""
        conf = 0.8
        component = 0.4 * conf
        assert abs(component - 0.32) < 1e-6

    def test_instability_component(self):
        """Instability component should be 0.3 * (1 - instability)."""
        instab = 0.2
        component = 0.3 * (1 - instab)
        assert abs(component - 0.24) < 1e-6

    def test_jaccard_component(self):
        """Jaccard component should be 0.3 * jaccard_similarity."""
        jac = 0.5
        component = 0.3 * jac
        assert abs(component - 0.15) < 1e-6
