import { afterEach, describe, expect, it } from "vitest";
import { _setMintTokenForTest, mintShareToken } from "@/lib/share/token";

describe("mintShareToken", () => {
  afterEach(() => {
    _setMintTokenForTest(null);
  });

  it("returns 64 lowercase hex chars (32 bytes) by default", () => {
    const tok = mintShareToken();
    expect(tok).toMatch(/^[0-9a-f]{64}$/);
  });

  it("produces unique tokens across many mints", () => {
    const seen = new Set<string>();
    for (let i = 0; i < 1000; i++) seen.add(mintShareToken());
    expect(seen.size).toBe(1000);
  });

  it("uses the test override when set, then restores real randomness on null", () => {
    _setMintTokenForTest(() => "tok_fixed");
    expect(mintShareToken()).toBe("tok_fixed");
    expect(mintShareToken()).toBe("tok_fixed");

    _setMintTokenForTest(null);
    const real = mintShareToken();
    expect(real).not.toBe("tok_fixed");
    expect(real).toMatch(/^[0-9a-f]{64}$/);
  });
});
