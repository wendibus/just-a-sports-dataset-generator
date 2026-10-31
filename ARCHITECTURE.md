# Architecture

## 1. System context

Just a Sports Dataset Generator turns a versioned dataset recipe into rendered media, automatic annotations, validation reports, and a provenance manifest. The core must run on Windows, macOS, and Linux while external render engines and models remain replaceable adapters.

## 2. Core pipeline

`recipe -> validation -> deterministic job plan -> scene graph -> render passes -> annotations -> validation -> export + manifest`

### Recipe and planning

Recipes use versioned JSON/YAML schemas. Validation resolves defaults, rejects unknown incompatible fields, records asset/model/plugin constraints, and expands a master seed into stable per-scene and per-frame seeds. Job plans are immutable and resumable.

### Scene generation

The scene graph represents venues, surfaces, athletes, skeletons, equipment, teams, materials, motion, cameras, lights, weather, and visibility relationships. Sports plugins provide rules and domain constraints without depending on a renderer.

### Render backends

A backend consumes a scene graph and produces RGB plus requested auxiliary passes such as instance IDs, semantic classes, depth, normals, motion vectors, and visibility. Backends declare capabilities, versions, supported platforms, and deterministic limitations.

### Annotation pipeline

Annotations are derived from scene truth, camera calibration, and render passes. Each exporter consumes one normalized annotation model. Validators check finite coordinates, image bounds, mask/box consistency, stable IDs, keypoint visibility, and cross-frame continuity.

### Export and provenance

Exporters write COCO, YOLO, MOT, or the native lossless format. A run manifest records normalized recipe, seeds, app/plugin/backend/model/schema versions, asset IDs and licenses, platform/hardware information, checksums, splits, and validation results.

## 3. Plugin model

Plugin types include sports, render backends, asset providers, motion generators, annotation exporters, and optional model integrations. Every plugin ships metadata containing:

- plugin ID and semantic version;
- plugin API version range;
- compatible application version range;
- supported OS and CPU architecture;
- required capabilities and dependencies;
- schema versions and migrations;
- artifact hashes and signature information.

Discovery must not execute untrusted plugin code. Validate metadata first, isolate plugins where practical, restrict filesystem/network access, and fail closed on incompatibility.

## 4. Data and compatibility contracts

Public contracts are versioned independently:

- application version;
- recipe and native dataset schema versions;
- plugin API version;
- renderer capability profile;
- exporter format profile;
- model metadata/API version.

Backward-compatible additions increment minor versions. Breaking changes require a major version, explicit migration, and fixtures proving old projects are either migrated safely or rejected with an actionable message.

## 5. Secure updater

### Policy

- channel: `stable`;
- automatic application updates: enabled;
- automatic compatible plugin updates: enabled;
- automatic model downloads: disabled;
- rollback: enabled.

### GitHub Releases layout

Each release publishes platform-specific application artifacts, a canonical manifest, a detached manifest signature, checksums, release notes, and optional plugin index updates. Stable clients only consume non-prerelease releases from the configured canonical repository.

### Manifest

The signed manifest should contain at least:

`schema_version`, `release_id`, `version`, `channel`, `published_at`, `minimum_updater_version`, artifacts keyed by OS/architecture, download URLs, sizes, SHA-256 hashes, signing key ID, application/plugin API ranges, schema ranges, migration requirements, plugin compatibility, model compatibility, and rollback metadata.

Canonicalize the manifest before signature verification. Trust is anchored in one or more public keys embedded with the application. Key rotation requires a transition signed by an already trusted key; revocations ship in a trusted manifest.

### Update flow

1. Fetch the release manifest over HTTPS with timeouts and size limits.
2. Verify repository/channel, signature, schema, freshness, and monotonic version policy.
3. Select exactly one artifact for the current OS and architecture.
4. Check application, updater, plugin API, schema, installed-plugin, and model compatibility.
5. Download to a staging directory; enforce declared size and SHA-256.
6. Validate archive paths, package signature where available, and executable layout.
7. Stop or coordinate with the running application.
8. Install using an atomic platform-appropriate swap.
9. Launch a bounded health check and record the version as last known good.
10. On failure, restore the retained version and preserve diagnostic logs.

Never execute code from an unverified package. Reject path traversal, links escaping staging, ambiguous platform matches, downgrade attempts unless explicitly authorized, expired/revoked keys, and manifests that would orphan required plugins or unreadably migrate user data.

### Platform installation

- Windows: signed installer or helper process; replace files outside the running process; recover from interrupted install.
- macOS: notarized/signed application bundle; verify signature and designated requirement before swap.
- Linux: prefer package-manager integration when installed from a system package; otherwise use an atomic versioned application directory and symlink/launcher switch.

User projects and configuration live outside the application directory and are never removed by rollback.

### Plugins and models

Plugin updates use the same signature, hash, staging, compatibility, health-check, and rollback principles. Auto-update only when the installed application's API and every declared dependency are compatible.

Models are never downloaded automatically. The UI may report that an optional or required model is available, but installation requires explicit consent with size, license, source, version, and compatibility shown beforehand. Model artifacts are hashed and recorded in dataset provenance.

## 6. Failure handling and observability

Operations emit structured local events without secrets or user dataset contents. Each job and update has a correlation ID. Interrupted generation resumes from completed immutable jobs. Update failures retain the previous version, a sanitized log, and a human-readable recovery action.

## 7. Security boundaries

Treat recipes, assets, plugins, manifests, archives, model files, and dataset paths as untrusted input. Apply schema and size limits, safe archive extraction, canonical paths, least privilege, network allowlists for updater endpoints, and explicit consent for external services. Secrets belong in platform credential storage, never recipes or manifests.

## 8. Testing strategy

- unit and property tests for seeds, transforms, visibility, validators, and compatibility resolution;
- contract tests shared by every plugin/backend/exporter;
- golden scenes with small checked-in expected annotations;
- cross-platform deterministic replay within documented tolerances;
- adversarial manifest/archive/plugin tests;
- fault injection at every updater transition;
- migration and rollback tests across supported version edges.

## 9. Initial milestones

1. Versioned recipe schema, normalized domain model, deterministic planner, and CLI.
2. Minimal reference renderer adapter with RGB and instance-ID output.
3. Native annotation model, COCO exporter, validators, and provenance manifest.
4. Plugin API and one sport reference plugin.
5. Desktop shell and job monitoring.
6. Signed release manifest tooling, compatibility resolver, atomic updater, and rollback.
