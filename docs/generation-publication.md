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
and validate all algorithm-native files there, sync every regular file and
directory in the staging tree, rename the staging directory to an immutable
generation ID, sync the generations directory, then atomically replace
`<root>/CURRENT` with the generation ID and sync the root. Publication rejects
symlinks and unsupported filesystem entries, including entries created through
the raw staging-directory accessor. Readers resolve only `CURRENT`, validate
that it names a safe generation directory, and then open files within that
directory.

Each newly published generation also contains a reserved `GENERATION.json`
inventory. It records version `1` and a sorted list of every regular file
other than the inventory itself, with its relative path, byte length, and
CRC32. `persistence::generation::verify_generation` strictly checks that
inventory, including the file set, lengths, and checksums;
`verify_current` resolves `CURRENT` and applies the same check. These are
explicit integrity checks rather than an implicit change to every reader.
Generations created before the inventory was introduced remain loadable through
`open_current`, but intentionally fail strict verification until migrated or
re-published.

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
- A pre-commit publication failure leaves the prior `CURRENT` value untouched.
- Publication takes a root-scoped single-writer lock; concurrent publishers
  fail explicitly rather than silently racing last-writer-wins.
- `publish_with_outcome` distinguishes a pre-commit error from a
  `DurabilityUncertain` result after `CURRENT` has been replaced but the final
  root sync failed. The latter is visible to readers and must be reconciled by
  the caller using the returned generation path and error.
- Readers never resolve a generation outside the configured root.
- Cleanup is a separate, retention-aware operation.

Generation cleanup is not yet safe to automate. `open_current` remains a
compatibility path without a reader lifetime token, while
`open_current_pinned` now provides a kernel-held per-generation shared lease.
Before retention or garbage collection is enabled, file-backed readers must
retain that lease and cleanup must acquire the corresponding exclusive lock
before removing a generation. Until reader adoption lands, published
generations are retained and cleanup remains an operator-owned decision.

The lock pathname is permanent, but ownership is held by a kernel advisory lock
on its open file handle. A process crash releases the lock without deleting the
pathname. Participants must cooperate with the lock, and remote filesystems
such as NFS/SMB require separate deployment validation; they are not part of the
default durability claim.

## Adoption gates

An index format may adopt the foundation only after it has:

1. a versioned manifest naming every required component;
2. validation before publication and on open;
3. interruption tests before and after pointer replacement, including handling
   of `DurabilityUncertain`;
4. a compatibility policy for old generations; and
5. a retention/garbage-collection policy.

IVF-PQ is the first candidate because it already has a versioned manifest,
file-backed readers, and component-level validation. Existing direct-directory
save APIs remain component-atomic until that migration lands.
