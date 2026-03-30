import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render } from "@testing-library/react";

import { CompassRose } from "../components/fire-details/CompassRose";

afterEach(cleanup);

describe("CompassRose", () => {
  it("renders an SVG element", () => {
    const { container } = render(<CompassRose directionDeg={230} />);
    const svg = container.querySelector("svg");
    expect(svg).not.toBeNull();
  });

  it("defaults to 40px size", () => {
    const { container } = render(<CompassRose directionDeg={0} />);
    const svg = container.querySelector("svg");
    expect(svg?.getAttribute("width")).toBe("40");
    expect(svg?.getAttribute("height")).toBe("40");
  });

  it("accepts a custom size prop", () => {
    const { container } = render(<CompassRose directionDeg={90} size={28} />);
    const svg = container.querySelector("svg");
    expect(svg?.getAttribute("width")).toBe("28");
    expect(svg?.getAttribute("height")).toBe("28");
  });

  it("renders all four cardinal labels", () => {
    const { getByText } = render(<CompassRose directionDeg={0} />);
    expect(getByText("N")).toBeDefined();
    expect(getByText("E")).toBeDefined();
    expect(getByText("S")).toBeDefined();
    expect(getByText("W")).toBeDefined();
  });

  it("sets aria-label with the direction", () => {
    const { container } = render(<CompassRose directionDeg={45} />);
    const svg = container.querySelector("svg");
    expect(svg?.getAttribute("aria-label")).toContain("45");
  });

  it("applies rotation via transform attribute on the arrow group", () => {
    const { container } = render(<CompassRose directionDeg={180} />);
    const groups = container.querySelectorAll("g[transform]");
    const rotated = Array.from(groups).some((g) =>
      g.getAttribute("transform")?.includes("rotate(180")
    );
    expect(rotated).toBe(true);
  });

  it("normalises direction > 360 correctly", () => {
    const { container } = render(<CompassRose directionDeg={400} />);
    const groups = container.querySelectorAll("g[transform]");
    // 400 % 360 = 40
    const rotated = Array.from(groups).some((g) =>
      g.getAttribute("transform")?.includes("rotate(40")
    );
    expect(rotated).toBe(true);
  });
});
