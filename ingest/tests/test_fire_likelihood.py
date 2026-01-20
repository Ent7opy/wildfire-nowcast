"""Tests for fire likelihood scoring logic."""

import pytest

from api.fires.service import (
    normalize_firms_confidence,
    compute_confidence_prior,
)


class TestNormalizeFirmsConfidence:
    """Test confidence normalization for different sensors and values."""
    
    def test_missing_confidence_returns_neutral(self):
        """Missing confidence should return neutral prior (0.5)."""
        assert normalize_firms_confidence(None, "VIIRS") == 0.5
        assert normalize_firms_confidence(None, "Terra") == 0.5
        assert normalize_firms_confidence(None, None) == 0.5
    
    def test_zero_confidence(self):
        """Zero confidence should normalize to 0."""
        assert normalize_firms_confidence(0.0, "VIIRS") == 0.0
        assert normalize_firms_confidence(0.0, "Terra") == 0.0
    
    def test_low_confidence_modis(self):
        """MODIS low confidence (10) should normalize to 0.1."""
        result = normalize_firms_confidence(10.0, "Terra")
        assert result == pytest.approx(0.1, abs=0.01)
        
        result = normalize_firms_confidence(10.0, "Aqua")
        assert result == pytest.approx(0.1, abs=0.01)
    
    def test_nominal_confidence_modis(self):
        """MODIS nominal confidence (50) should normalize to 0.5."""
        result = normalize_firms_confidence(50.0, "Terra")
        assert result == pytest.approx(0.5, abs=0.01)
    
    def test_high_confidence_modis(self):
        """MODIS high confidence (90) should normalize to 0.9."""
        result = normalize_firms_confidence(90.0, "Terra")
        assert result == pytest.approx(0.9, abs=0.01)
    
    def test_viirs_low_confidence(self):
        """VIIRS low confidence range (0-30) should normalize proportionally."""
        assert normalize_firms_confidence(0.0, "VIIRS") == 0.0
        assert normalize_firms_confidence(15.0, "VIIRS") == pytest.approx(0.15, abs=0.01)
        assert normalize_firms_confidence(30.0, "VIIRS") == pytest.approx(0.3, abs=0.01)
    
    def test_viirs_nominal_confidence(self):
        """VIIRS nominal confidence range (30-70) should normalize proportionally."""
        assert normalize_firms_confidence(50.0, "VIIRS") == pytest.approx(0.5, abs=0.01)
        assert normalize_firms_confidence(70.0, "VIIRS") == pytest.approx(0.7, abs=0.01)
    
    def test_viirs_high_confidence(self):
        """VIIRS high confidence range (70-100) should normalize proportionally."""
        assert normalize_firms_confidence(85.0, "VIIRS") == pytest.approx(0.85, abs=0.01)
        assert normalize_firms_confidence(100.0, "VIIRS") == pytest.approx(1.0, abs=0.01)
    
    def test_confidence_clamping_above_range(self):
        """Confidence values above 100 should be clamped to 100."""
        assert normalize_firms_confidence(150.0, "VIIRS") == 1.0
        assert normalize_firms_confidence(200.0, "Terra") == 1.0
    
    def test_confidence_clamping_below_range(self):
        """Confidence values below 0 should be clamped to 0."""
        assert normalize_firms_confidence(-10.0, "VIIRS") == 0.0
        assert normalize_firms_confidence(-50.0, "Terra") == 0.0
    
    def test_sensor_agnostic_normalization(self):
        """Same confidence value should normalize identically across sensors."""
        # Both MODIS and VIIRS use 0-100 scale, so normalization should be identical
        assert (
            normalize_firms_confidence(50.0, "VIIRS") 
            == normalize_firms_confidence(50.0, "Terra")
        )
        assert (
            normalize_firms_confidence(90.0, "VIIRS") 
            == normalize_firms_confidence(90.0, "Aqua")
        )
    
    def test_unknown_sensor(self):
        """Unknown sensor should still normalize correctly."""
        assert normalize_firms_confidence(50.0, "UnknownSensor") == pytest.approx(0.5, abs=0.01)
        assert normalize_firms_confidence(75.0, None) == pytest.approx(0.75, abs=0.01)


class TestComputeConfidencePrior:
    """Test confidence prior computation with weight constraints."""
    
    def test_default_max_weight(self):
        """Default max weight should be 0.2 (20%)."""
        # High confidence (100) with max weight 0.2 should give 0.2
        result = compute_confidence_prior(100.0, "VIIRS")
        assert result == pytest.approx(0.2, abs=0.01)
    
    def test_high_confidence_capped_at_max_weight(self):
        """High confidence should be capped at max_weight."""
        result = compute_confidence_prior(100.0, "VIIRS", max_weight=0.2)
        assert result == pytest.approx(0.2, abs=0.01)
        
        result = compute_confidence_prior(90.0, "Terra", max_weight=0.2)
        assert result == pytest.approx(0.18, abs=0.01)  # 0.9 * 0.2
    
    def test_low_confidence_scaled_proportionally(self):
        """Low confidence should be scaled proportionally."""
        result = compute_confidence_prior(10.0, "Terra", max_weight=0.2)
        assert result == pytest.approx(0.02, abs=0.01)  # 0.1 * 0.2
        
        result = compute_confidence_prior(50.0, "VIIRS", max_weight=0.2)
        assert result == pytest.approx(0.1, abs=0.01)  # 0.5 * 0.2
    
    def test_missing_confidence_neutral_prior(self):
        """Missing confidence should give neutral prior (0.5 * max_weight)."""
        result = compute_confidence_prior(None, "VIIRS", max_weight=0.2)
        assert result == pytest.approx(0.1, abs=0.01)  # 0.5 * 0.2
    
    def test_custom_max_weight(self):
        """Custom max_weight should be respected."""
        result = compute_confidence_prior(100.0, "VIIRS", max_weight=0.15)
        assert result == pytest.approx(0.15, abs=0.01)
        
        result = compute_confidence_prior(100.0, "VIIRS", max_weight=0.3)
        assert result == pytest.approx(0.3, abs=0.01)
    
    def test_confidence_alone_cannot_dominate(self):
        """Even perfect confidence should contribute at most max_weight."""
        # This ensures confidence alone cannot mark a detection as a real fire
        result = compute_confidence_prior(100.0, "VIIRS", max_weight=0.2)
        assert result <= 0.2
        
        # Even with generous max_weight, it's still bounded
        result = compute_confidence_prior(100.0, "VIIRS", max_weight=0.5)
        assert result <= 0.5
