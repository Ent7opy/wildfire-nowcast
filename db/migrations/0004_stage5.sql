-- Stage 5 — A' pivot: Clerk auth + per-user AOIs.
--
-- Authoritative source: docs/pivot-architecture.md §3.1 (users.id is the
-- Clerk user_id directly, no schema widening required); pm/briefs/19-stage5-clerk-auth.md.
--
-- The Stage 1 `users` table already has `id TEXT PRIMARY KEY`, which slots
-- in Clerk-issued `user_xxx` ids unchanged. Stage 5's only DB change is to
-- drop the seeded single-user stub. Cascading FKs on `aois.user_id` will
-- remove any AOIs that were attached to the stub during pre-Stage-5 dev.

DELETE FROM "users" WHERE "id" = 'stub-user-1';
