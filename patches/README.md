# Runtime compatibility patches

Patch files in this directory are never applied to `third_party/`. They document
reproducible, platform-specific changes made only to an installed environment after
the upstream source and exact package version have been verified.

## robosuite 1.4.0 Windows WGL

`robosuite-1.4.0-windows-wgl.patch` implements the one-line Windows renderer change
described by the official robosuite Windows installation documentation. Upstream
v1.4.0 otherwise forces `egl` on Windows and then rejects it as invalid. Apply from
the active environment's `Lib/site-packages` directory with `git apply`.
