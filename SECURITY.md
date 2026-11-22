# Security policy

## Supported versions

gregorithm is in its 0.x series. Security fixes are applied to the latest release only.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's **Report a vulnerability** button in the
repository's *Security* tab (private vulnerability reporting). Do not open a public issue, publish
details or attach private battery data or credentials.

Include the affected version, platform, impact, reproduction steps and, if possible, a minimal
synthetic input. Reports are handled on a best-effort basis; you will receive an acknowledgement
and be kept informed about the fix.

## Scope

gregorithm is an offline library. It must not access the network, send telemetry, import GUI
toolkits, follow symbolic links out of a result store or write outside the caller-provided result
root and export destination. Please report any behaviour contrary to this.
