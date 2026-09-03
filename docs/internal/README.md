# Internal and historical notes

The files in this directory capture historical design decisions, operational
run notes, and review findings from earlier stages of the project. They are kept
for context, but the names, identifiers, and infrastructure details have been
replaced with neutral example values for the open-source release.

- `legacy-system-study.md`: what was learned from the prior internal system and
  what was intentionally decoupled.
- `review-reservations.md`: design reservations raised during autonomous
  fleet-management review passes and the budget-reservation subsystem design.

Do not commit operational logs, output images, provider keys, or hostnames to
this repository. The `outputs/` and runtime log paths are ignored in
`.gitignore`.
