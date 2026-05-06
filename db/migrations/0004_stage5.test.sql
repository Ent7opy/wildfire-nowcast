-- Stage 5 — PGlite-compatible variant of 0004_stage5.sql.
--
-- Identical to the production DDL: a single DELETE has no PostGIS
-- dependencies. Kept as a separate file so the PGlite harness applies
-- migrations in the same numbered order as Neon.

DELETE FROM "users" WHERE "id" = 'stub-user-1';
