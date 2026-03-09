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
    """Poll JIT forecast job status and advance lifecycle state.

    This component polls GET /forecast/jit/{job_id} every 2 seconds until
    the job reaches a terminal state (completed or failed).

    On completion, updates state with forecast run_id and triggers
    a map refresh.

    Each script execution performs exactly one poll iteration. Streamlit's
    st.rerun() triggers the next poll after a 2-second delay.
    """
    fj = app_state.forecast_job

    max_polls = 300  # 10 minutes max (300 * 2s)

    # Check timeout at START before making API call
    if fj.poll_count >= max_polls:
        fj.notification = {
            "kind": "error",
            "message": "Forecast timed out after 10 minutes.",
            "created_at": time.time(),
            "ttl_seconds": 45.0,
        }
        fj.clear()
        app_state._persist()
        st.rerun()
        return

    # Perform ONE status check per script execution
    try:
        status_data = get_jit_forecast_status(job_id)
        status = status_data.get("status", "unknown")

        if status == "completed":
            result = status_data.get("result", {})
            run_id = result.get("run_id")

            if run_id:
                fj.complete(run_id, job_id)
                app_state.layers.show_forecast = True
                location_label = str(
                    (fj.last_forecast or {}).get("location_label") or "the selected area"
                )
                fj.notification = {
                    "kind": "ready",
                    "message": f"Forecast for the fire event from {location_label} is ready!",
                    "created_at": time.time(),
                    "ttl_seconds": 600.0,
                    "run_id": run_id,
                    "target": {
                        "lat": (fj.last_forecast or {}).get("lat"),
                        "lon": (fj.last_forecast or {}).get("lon"),
                        "event_snapshot": (fj.last_forecast or {}).get("event_snapshot"),
                        "event_id": (fj.last_forecast or {}).get("event_id"),
                        "event_key": (fj.last_forecast or {}).get("event_key"),
                    },
                }
            else:
                fj.clear()

            app_state._persist()
            st.rerun()

        elif status == "failed":
            error_msg = status_data.get("error", "Unknown error")
            fj.notification = {
                "kind": "error",
                "message": f"Forecast failed: {error_msg}",
                "created_at": time.time(),
                "ttl_seconds": 45.0,
            }
            fj.clear()
            app_state._persist()
            st.rerun()
            return

        else:
            # In-progress — poll quietly; user feedback comes from notifications.
            fj.increment_poll()
            app_state._persist()
            time.sleep(2)
            st.rerun()

    except ApiUnavailableError:
        # Transient backend hiccup — retry quietly to avoid noisy UI.
        fj.increment_poll()
        app_state._persist()
        time.sleep(2)
        st.rerun()

    except ApiError as e:
        msg = "Job not found. It may have expired." if e.status_code == 404 else f"Error checking job status: {e.message}"
        fj.notification = {
            "kind": "error",
            "message": msg,
            "created_at": time.time(),
            "ttl_seconds": 45.0,
        }

        fj.clear()
        app_state._persist()
        st.rerun()
