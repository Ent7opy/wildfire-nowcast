# Task: Extract weather point lookup into a shared module

The function that fetches GFS weather data for a lat/lon point currently lives inside the fire scoring module. It needs to be accessible from other parts of the API without creating a circular import.

Move it to a shared location so it can be called from both the risk grid and the new fire detail endpoint. Make sure existing callers still work after the move.
