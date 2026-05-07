/**
 * Drizzle's `db.execute(sql\`...\`)` returns different shapes per backend:
 *   - node-postgres (Neon): `{ rows: T[]; rowCount: number; ... }` (pg.QueryResult)
 *   - PGlite:               `T[]` directly (or sometimes the same `{ rows }` shape)
 *
 * Every call site in this directory needs the same defensive unwrap. This
 * helper isolates the unsafe cast in one typed signature instead of repeating
 * `(result.rows ?? (result as unknown as T[])) as T[]` at each call.
 *
 * Trust note: callers assert the row shape `T`; we cannot validate it at
 * runtime without paying for a Zod parse on every query. Same trust boundary
 * as before — just expressed once.
 */

export function decodeRows<T>(result: unknown): T[] {
  if (result == null) return [];
  if (Array.isArray(result)) return result as T[];
  if (typeof result === "object" && "rows" in result) {
    const rows = (result as { rows?: unknown }).rows;
    if (Array.isArray(rows)) return rows as T[];
  }
  return [];
}

/**
 * DELETE/UPDATE row-count extraction. node-postgres exposes `rowCount`;
 * PGlite exposes `affectedRows`. Returns 0 if neither is present (the query
 * succeeded but the driver did not report a count).
 */
export function decodeRowCount(result: unknown): number {
  if (result == null || typeof result !== "object") return 0;
  const r = result as { rowCount?: unknown; affectedRows?: unknown };
  if (typeof r.rowCount === "number") return r.rowCount;
  if (typeof r.affectedRows === "number") return r.affectedRows;
  return 0;
}
