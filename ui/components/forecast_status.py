"""Forecast status polling component for JIT forecast jobs."""

import time

import streamlit as st

from state import app_state
from api_client import (
    ApiError,
    ApiUnavailableError,
    get_jit_forecast_status,
)


def render_forecast_status_polling(job_id: str) -> None:
    """Poll JIT forecast job status and display progress.

    This component polls GET /forecast/jit/{job_id} every 2 seconds until
    the job reaches a terminal state (completed or failed).

    On completion, updates state with forecast run_id and triggers
    a map refresh.

    Each script execution performs exactly one poll iteration. Streamlit's
    st.rerun() triggers the next poll after a 2-second delay.
    """
    placeholder = st.empty()
    fj = app_state.forecast_job

    max_polls = 300  # 10 minutes max (300 * 2s)

    # Check timeout at START before making API call
    if fj.poll_count >= max_polls:
        placeholder.error("Forecast timed out after 10 minutes.")
        fj.clear()
        app_state._persist()
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

            if run_id:
                fj.complete(run_id, job_id)
                app_state.layers.show_forecast = True
            else:
                fj.clear()

            app_state._persist()
            st.rerun()

        elif status == "failed":
            error_msg = status_data.get("error", "Unknown error")
            placeholder.error(f"Forecast failed: {error_msg}")
            fj.clear()
            app_state._persist()
            return

        else:
            # In-progress — display and schedule next poll
            placeholder.info(f"{progress_message} (Status: {status})")
            fj.increment_poll()
            app_state._persist()
            time.sleep(2)
            st.rerun()

    except ApiUnavailableError:
        placeholder.warning("API temporarily unavailable. Retrying...")
        fj.increment_poll()
        app_state._persist()
        time.sleep(2)
        st.rerun()

    except ApiError as e:
        if e.status_code == 404:
            placeholder.error("Job not found. It may have expired.")
        else:
            placeholder.error(f"Error checking job status: {e.message}")

        fj.clear()
        app_state._persist()
