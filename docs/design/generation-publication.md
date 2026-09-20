# Design: generation-based snapshot publication

Status: implemented foundation
Scope: multi-file persistence publication for native index layouts
Decision: use a staged generation directory and one atomic `CURRENT` pointer

## Problem

Index snapshots are naturally multi-file: graph metadata, vectors, codes,
manifests, and derived sidecars have different access patterns. Replacing those
files independently can expose a reader to a mixed snapshot after interruption
or concurrent publication.

## Chosen approach

Writers create a hidden staging directory beneath `<root>/generations`, write
and validate all algorithm-native files there, sync the files and staging
directory, rename the staging directory to an immutable generation ID, sync the
generations directory, then atomically replace `<root>/CURRENT` with the
generation ID. Readers resolve only `CURRENT`, validate that it names a safe
generation directory, and then open files within that directory.

The first implementation is a reusable filesystem foundation in
`persistence::generation`. It deliberately does not rewrite existing index
formats automatically. Each format can adopt it once its manifest and
validation path are ready.

## Non-goals

- WAL or update durability; `segstore`/`durability` remain responsible for that.
- A universal container format or forced layout shared by all algorithms.
- Deleting old generations during publication.
- Claiming power-loss safety for formats that have not adopted this protocol.
- Cross-host or object-store publication semantics.

## Invariants

- `CURRENT` contains one basename only; absolute paths and traversal components
  are rejected.
- A published generation is never modified by the publisher after the pointer
  changes.
- A failed publication leaves the prior `CURRENT` value untouched.
- Readers never resolve a generation outside the configured root.
- Cleanup is a separate, retention-aware operation.

## Adoption gates

An index format may adopt the foundation only after it has:

1. a versioned manifest naming every required component;
2. validation before publication and on open;
3. interruption tests before and after pointer replacement;
4. a compatibility policy for old generations; and
5. a retention/garbage-collection policy.

IVF-PQ is the first candidate because it already has a versioned manifest,
file-backed readers, and component-level validation. Existing direct-directory
save APIs remain component-atomic until that migration lands.
