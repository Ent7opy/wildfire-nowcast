/**
 * Minimal RFC-4180 CSV escaping. Wrap in quotes when the cell contains
 * `"`, `,`, `\n`, or `\r`; double internal quotes.
 */
export function escapeCsvCell(value: unknown): string {
  if (value === null || value === undefined) return "";
  const s = String(value);
  if (/[",\r\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

export function csvRow(cells: unknown[]): string {
  return cells.map(escapeCsvCell).join(",");
}
