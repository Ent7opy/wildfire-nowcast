export function forecastButtonState(args: {
  forecastRunning: boolean;
  sameEventCompleted: boolean;
}): { label: string; disabled: boolean; reason: string } {
  if (args.forecastRunning) {
    return {
      label: "Generating Spread Forecast...",
      disabled: true,
      reason: "A forecast is already running."
    };
  }
  if (args.sameEventCompleted) {
    return {
      label: "Spread Forecast Already Generated",
      disabled: true,
      reason: "A forecast already exists for this fire event."
    };
  }
  return {
    label: "Generate Spread Forecast",
    disabled: false,
    reason: ""
  };
}
