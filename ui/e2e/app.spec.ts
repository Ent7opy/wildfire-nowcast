import { expect, test } from "@playwright/test";

test("loads app shell and map panel", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByText("Wildfire Nowcast & Forecast")).toBeVisible();
  await expect(page.getByText("Map")).toBeVisible();
});
