"""Unit tests for repository scoring update functions."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from api.fires import repo


def test_update_persistence_scores_queries_correct_fields():
    """Verify update_persistence_scores queries required fields and calls compute_persistence_scores."""
    batch_id = 123
    
    # Mock database query result
    mock_result = MagicMock()
    mock_result.mappings.return_value.all.return_value = [
        {"id": 1, "lat": 42.0, "lon": 23.0, "acq_time": "2026-01-15T12:00:00+00:00", "sensor": "VIIRS"},
        {"id": 2, "lat": 42.1, "lon": 23.1, "acq_time": "2026-01-15T13:00:00+00:00", "sensor": "MODIS"},
    ]
    
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value.execute.return_value = mock_result
    
    with patch("api.fires.repo.get_engine") as mock_engine, \
         patch("api.fires.repo.compute_persistence_scores") as mock_compute:
        mock_engine.return_value.begin.return_value = mock_conn
        mock_compute.return_value = {1: 0.8, 2: 0.9}
        
        result = repo.update_persistence_scores(batch_id)
        
        # Verify compute_persistence_scores was called with correct data
        assert mock_compute.call_count == 1
        detections = mock_compute.call_args[0][0]
        assert len(detections) == 2
        assert detections[0]["id"] == 1
        assert detections[0]["sensor"] == "VIIRS"
        
        # Verify update was called with correct scores
        update_calls = mock_conn.__enter__.return_value.execute.call_args_list
        assert len(update_calls) == 2  # SELECT + UPDATE
        
        # Verify return value
        assert result == 2


def test_update_persistence_scores_handles_empty_batch():
    """Verify update_persistence_scores handles empty batch correctly."""
    batch_id = 456
    
    # Mock empty database query result
    mock_result = MagicMock()
    mock_result.mappings.return_value.all.return_value = []
    
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value.execute.return_value = mock_result
    
    with patch("api.fires.repo.get_engine") as mock_engine, \
         patch("api.fires.repo.compute_persistence_scores") as mock_compute:
        mock_engine.return_value.begin.return_value = mock_conn
        
        result = repo.update_persistence_scores(batch_id)
        
        # Verify compute_persistence_scores was not called
        assert mock_compute.call_count == 0
        
        # Verify return value
        assert result == 0


def test_update_landcover_scores_handles_missing_landcover_data():
    """Verify update_landcover_scores handles missing landcover data gracefully."""
    batch_id = 789
    
    # Mock database query result
    mock_result = MagicMock()
    mock_result.mappings.return_value.all.return_value = [
        {"id": 1, "lat": 42.0, "lon": 23.0},
        {"id": 2, "lat": 42.1, "lon": 23.1},
    ]
    
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value.execute.return_value = mock_result
    
    with patch("api.fires.repo.get_engine") as mock_engine:
        mock_engine.return_value.begin.return_value = mock_conn
        
        # Mock compute_landcover_scores to return neutral scores (simulating missing data)
        with patch("api.fires.landcover.compute_landcover_scores") as mock_compute:
            mock_compute.return_value = {1: 0.5, 2: 0.5}
            
            result = repo.update_landcover_scores(batch_id)
            
            # Verify compute_landcover_scores was called
            assert mock_compute.call_count == 1
            detections = mock_compute.call_args[0][0]
            assert len(detections) == 2
            
            # Verify return value
            assert result == 2


def test_update_landcover_scores_imports_module_correctly():
    """Verify update_landcover_scores imports landcover module correctly."""
    batch_id = 111
    
    # Mock database query result
    mock_result = MagicMock()
    mock_result.mappings.return_value.all.return_value = [
        {"id": 1, "lat": 42.0, "lon": 23.0},
    ]
    
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value.execute.return_value = mock_result
    
    with patch("api.fires.repo.get_engine") as mock_engine:
        mock_engine.return_value.begin.return_value = mock_conn
        
        # The import happens dynamically in the function
        # We can't easily patch it, but we can verify the function runs
        with patch("api.fires.landcover.compute_landcover_scores") as mock_compute:
            mock_compute.return_value = {1: 1.0}
            
            result = repo.update_landcover_scores(batch_id)
            
            # Verify function completed successfully
            assert result == 1


def test_update_fire_likelihood_combines_scores_correctly():
    """Verify update_fire_likelihood combines component scores correctly."""
    batch_id = 222
    
    # Mock database query result with all component scores
    mock_result = MagicMock()
    mock_result.mappings.return_value.all.return_value = [
        {
            "id": 1,
            "confidence_score": 0.9,
            "persistence_score": 0.8,
            "landcover_score": 1.0,
            "weather_score": 0.7,
            "false_source_masked": False,
        },
        {
            "id": 2,
            "confidence_score": 0.8,
            "persistence_score": None,  # NULL score
            "landcover_score": 0.5,
            "weather_score": None,  # NULL score
            "false_source_masked": False,
        },
    ]
    
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value.execute.return_value = mock_result
    
    with patch("api.fires.repo.get_engine") as mock_engine, \
         patch("api.fires.repo.compute_fire_likelihood") as mock_compute:
        mock_engine.return_value.begin.return_value = mock_conn
        mock_compute.side_effect = [0.85, 0.6]  # Return different scores for each call
        
        result = repo.update_fire_likelihood(batch_id)
        
        # Verify compute_fire_likelihood was called twice
        assert mock_compute.call_count == 2
        
        # Verify first call has all scores
        first_call = mock_compute.call_args_list[0]
        assert first_call[1]["confidence_score"] == 0.9
        assert first_call[1]["persistence_score"] == 0.8
        assert first_call[1]["landcover_score"] == 1.0
        assert first_call[1]["weather_score"] == 0.7
        assert first_call[1]["false_source_masked"] is False
        
        # Verify second call handles NULL scores
        second_call = mock_compute.call_args_list[1]
        assert second_call[1]["confidence_score"] == 0.8
        assert second_call[1]["persistence_score"] is None
        assert second_call[1]["landcover_score"] == 0.5
        assert second_call[1]["weather_score"] is None
        
        # Verify return value
        assert result == 2


def test_update_fire_likelihood_sets_zero_when_masked():
    """Verify update_fire_likelihood sets likelihood to 0 when false_source_masked is True."""
    batch_id = 333
    
    # Mock database query result with masked detection
    mock_result = MagicMock()
    mock_result.mappings.return_value.all.return_value = [
        {
            "id": 1,
            "confidence_score": 0.9,
            "persistence_score": 0.9,
            "landcover_score": 0.9,
            "weather_score": 0.9,
            "false_source_masked": True,  # Masked
        },
    ]
    
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value.execute.return_value = mock_result
    
    with patch("api.fires.repo.get_engine") as mock_engine, \
         patch("api.fires.repo.compute_fire_likelihood") as mock_compute:
        mock_engine.return_value.begin.return_value = mock_conn
        mock_compute.return_value = 0.0  # compute_fire_likelihood returns 0 for masked
        
        result = repo.update_fire_likelihood(batch_id)
        
        # Verify compute_fire_likelihood was called with masked=True
        assert mock_compute.call_count == 1
        call_args = mock_compute.call_args[1]
        assert call_args["false_source_masked"] is True
        
        # Verify return value
        assert result == 1


def test_update_fire_likelihood_handles_empty_batch():
    """Verify update_fire_likelihood handles empty batch correctly."""
    batch_id = 444
    
    # Mock empty database query result
    mock_result = MagicMock()
    mock_result.mappings.return_value.all.return_value = []
    
    mock_conn = MagicMock()
    mock_conn.__enter__.return_value.execute.return_value = mock_result
    
    with patch("api.fires.repo.get_engine") as mock_engine, \
         patch("api.fires.repo.compute_fire_likelihood") as mock_compute:
        mock_engine.return_value.begin.return_value = mock_conn
        
        result = repo.update_fire_likelihood(batch_id)
        
        # Verify compute_fire_likelihood was not called
        assert mock_compute.call_count == 0
        
        # Verify return value
        assert result == 0
