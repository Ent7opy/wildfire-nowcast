"""Forecast status polling component for JIT forecast jobs."""

import time
from typing import Optional

import streamlit as st

from api_client import ApiError, ApiUnavailableError, get_jit_forecast_status


def render_forecast_status_polling(job_id: str) -> None:
    """Poll JIT forecast job status and display progress.
    
    This component polls GET /forecast/jit/{job_id} every 2 seconds until
    the job reaches a terminal state (completed or failed).
    
    On completion, updates session state with forecast run_id and triggers
    a map refresh.
    
    Each script execution performs exactly one poll iteration. Streamlit's
    st.rerun() triggers the next poll after a 2-second delay.
    """
    placeholder = st.empty()
    
    max_polls = 300  # 10 minutes max (300 * 2s)
    
    # Initialize or retrieve poll count from session state
    poll_count = st.session_state.setdefault("jit_poll_count", 0)
    
    # Check timeout at START before making API call
    if poll_count >= max_polls:
        placeholder.error("Forecast timed out after 10 minutes.")
        if "jit_job_id" in st.session_state:
            del st.session_state.jit_job_id
        if "jit_poll_count" in st.session_state:
            del st.session_state.jit_poll_count
        return
    
    # Perform ONE status check per script execution
    try:
        status_data = get_jit_forecast_status(job_id)
        status = status_data.get("status", "unknown")
        progress_message = status_data.get("progress_message", "Processing...")
        
        if status == "completed":
            result = status_data.get("result", {})
            run_id = result.get("run_id")
            
            placeholder.success(f"Forecast complete! Run ID: {run_id}")
            
            # Update session state with forecast result
            if run_id:
                st.session_state.last_forecast = {
                    "run": {"id": run_id},
                    "job_id": job_id,
                    "completed_at": time.time(),
                }
                st.session_state.show_forecast = True
            
            # Clear polling state to stop polling
            if "jit_job_id" in st.session_state:
                del st.session_state.jit_job_id
            if "jit_poll_count" in st.session_state:
                del st.session_state.jit_poll_count
            
            # Trigger rerun to refresh map
            st.rerun()
            
        elif status == "failed":
            error_msg = status_data.get("error", "Unknown error")
            placeholder.error(f"Forecast failed: {error_msg}")
            
            # Clear polling state to stop polling
            if "jit_job_id" in st.session_state:
                del st.session_state.jit_job_id
            if "jit_poll_count" in st.session_state:
                del st.session_state.jit_poll_count
            return
            
        else:
            # In-progress status - display and schedule next poll
            placeholder.info(f"{progress_message} (Status: {status})")
            st.session_state.jit_poll_count += 1
            time.sleep(2)
            st.rerun()
            
    except ApiUnavailableError:
        placeholder.warning("API temporarily unavailable. Retrying...")
        st.session_state.jit_poll_count += 1
        time.sleep(2)
        st.rerun()
        
    except ApiError as e:
        if e.status_code == 404:
            placeholder.error("Job not found. It may have expired.")
        else:
            placeholder.error(f"Error checking job status: {e.message}")
        
        # Clear polling state on error
        if "jit_job_id" in st.session_state:
            del st.session_state.jit_job_id
        if "jit_poll_count" in st.session_state:
            del st.session_state.jit_poll_count
