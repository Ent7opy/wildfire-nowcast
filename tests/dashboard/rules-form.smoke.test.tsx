/**
 * Smoke test: rules form renders without throwing. Static markup only.
 */
import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server.node";
import { RulesForm } from "@/app/dashboard/_components/rules-form";

describe("<RulesForm>", () => {
  it("renders defaults when initial is null", () => {
    const html = renderToStaticMarkup(
      <RulesForm aoiId="11111111-1111-1111-1111-111111111111" initial={null} />,
    );
    expect(html).toContain("Distance buffer");
    expect(html).toContain("Min confidence");
    expect(html).toContain("Min FRP");
    expect(html).toContain("Quiet hours");
    expect(html).toContain("Save rules");
  });

  it("renders provided initial values into inputs", () => {
    const html = renderToStaticMarkup(
      <RulesForm
        aoiId="11111111-1111-1111-1111-111111111111"
        initial={{
          distanceBufferKm: 50,
          minConfidence: "high",
          minFrpMw: 12.5,
          quietHours: { tz: "America/Los_Angeles", startHour: 22, endHour: 7 },
          pausedUntil: null,
          notifyChannels: [{ type: "email", target: "alice@example.com" }],
        }}
      />,
    );
    expect(html).toContain("alice@example.com");
    expect(html).toContain("America/Los_Angeles");
  });
});
