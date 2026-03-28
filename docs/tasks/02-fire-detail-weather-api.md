# Task: Add weather conditions to the fire detail API response

When a user fetches a fire detection's details, the response should include current weather conditions at that location — wind speed, wind direction, relative humidity, temperature, and 24h precipitation — drawn from the GFS data already stored in the system.

If no recent weather data covers the fire's location, return a null weather block with a reason rather than omitting the field or erroring. Include enough metadata in the response (data age, source resolution) for the UI to communicate data quality to the user.

Reference the spec at `docs/spec_meteoalarm_weather_warnings.md` for the expected response shape and acceptance criteria.
