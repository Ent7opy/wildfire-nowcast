/**
 * Stage 8 follow-up — pin `coversBucket` predicates for NIFC + CWFIS.
 *
 * Reviewer flagged that `coversBucket` uses rectangular bbox (not a hardcoded
 * prefix list), so polygons / buckets near the bbox edges deserve explicit
 * coverage. These are pure predicate tests; no network.
 */
import { describe, expect, it } from "vitest";
import { selectSourceForBucket, AUTHORITY_SOURCES } from "@/lib/ai/authority/sources";

describe("selectSourceForBucket", () => {
  it("selects NIFC for a CONUS bucket", () => {
    // California: lon -120, lat 38 → bucket SW corner W120_N35.
    const s = selectSourceForBucket("5x5:W120_N35");
    expect(s?.id).toBe("nifc");
  });

  it("selects NIFC for an Alaska bucket", () => {
    // Interior Alaska: lon -150, lat 65.
    const s = selectSourceForBucket("5x5:W150_N65");
    expect(s?.id).toBe("nifc");
  });

  it("selects CWFIS for a Canadian bucket outside the NIFC overlap", () => {
    // Northern Quebec: lon -75, lat 55 → not in CONUS bbox (lat>50), not Alaska.
    const s = selectSourceForBucket("5x5:W075_N55");
    expect(s?.id).toBe("cwfis");
  });

  it("prefers NIFC over CWFIS in the southern-Canada overlap band", () => {
    // Both bboxes include lon -100, lat 45 (NIFC CONUS lat 24..50, lon -125..-65;
    // CWFIS lat 41..83, lon -141..-52). NIFC is listed first in
    // AUTHORITY_SOURCES so it wins by registration order — pin that behavior.
    const s = selectSourceForBucket("5x5:W100_N45");
    expect(s?.id).toBe("nifc");
  });

  it("returns null for a bucket outside both coverage areas", () => {
    // Sydney, AU.
    expect(selectSourceForBucket("5x5:E150_S35")).toBeNull();
    // Mid-Atlantic ocean.
    expect(selectSourceForBucket("5x5:W030_N30")).toBeNull();
    // Continental Europe (Iberia) — ICNF gap, filed as a Vanyo blocker.
    expect(selectSourceForBucket("5x5:W010_N40")).toBeNull();
  });

  it("returns null for a malformed bucket key", () => {
    expect(selectSourceForBucket("not-a-bucket")).toBeNull();
    expect(selectSourceForBucket("5x5:Z125_N40")).toBeNull();
    expect(selectSourceForBucket("")).toBeNull();
  });

  it("exposes NIFC and CWFIS in registration order", () => {
    // Pins ordering: any future addition must not silently displace NIFC's
    // priority in the CONUS/Canada overlap band.
    expect(AUTHORITY_SOURCES.map((s) => s.id)).toEqual(["nifc", "cwfis"]);
  });
});
