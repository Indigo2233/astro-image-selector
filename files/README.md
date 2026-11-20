# astro-files

Recursive FITS discovery and safe project file operations.

The crate skips the project-root archive directory when scanning. Rejected
files are moved atomically where possible, with a copy-and-persist fallback
for filesystems that do not support hard links.
