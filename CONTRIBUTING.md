# Contributing

Run the workspace checks from the repository root:

```powershell
cargo fmt --all -- --check
cargo test --workspace --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
```

Keep algorithm changes covered by the existing regression fixtures. Release
binaries are built by GitHub Actions from version tags and are not committed
to source control.
