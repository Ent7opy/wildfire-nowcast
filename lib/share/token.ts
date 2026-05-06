/**
 * Share-token generator. 32 bytes of crypto randomness, hex-encoded.
 * Test-overridable via `_setMintTokenForTest` so deterministic-output tests
 * don't drift when they exercise the route handler end to end.
 */
import { randomBytes } from "node:crypto";

let testMint: (() => string) | null = null;

export function _setMintTokenForTest(fn: (() => string) | null): void {
  testMint = fn;
}

export function mintShareToken(): string {
  if (testMint) return testMint();
  return randomBytes(32).toString("hex");
}
