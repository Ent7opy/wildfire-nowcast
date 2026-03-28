# Task: Render weather conditions in the fire detail panel

When a user clicks a fire detection, the detail panel should show the weather block returned by the API — wind, humidity, temperature, precipitation. Wind direction should be human-readable (compass bearing or arrow), not a raw degree value. The data source and resolution should be visible so users don't over-interpret the precision.

When weather data is unavailable, show a short inline message explaining why — no blank sections, no silent omissions.

Reference the spec at `docs/spec_meteoalarm_weather_warnings.md` for what the API response looks like and what the null state should communicate.
