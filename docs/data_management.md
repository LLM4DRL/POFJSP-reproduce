# Data Management

This directory contains guidelines for managing large files in the POFJSP repository.

## Large File Policy

Files larger than 10MB should not be committed directly to the repository. Instead:

1. **Log Files**: Large training/performance logs are automatically compressed
2. **Data Files**: Use Git LFS for files like .parquet, .h5, .pkl
3. **Model Checkpoints**: Store in `checkpoints/` directory (gitignored)
4. **Archive**: Large files moved to `archive/large_files/` (gitignored)

## Compressed Logs

Training logs are automatically compressed using gzip:
- `training.log` -> `training.log.gz`
- Compression typically reduces size by 70-90%

## Git LFS Setup

For data files that must be versioned:

```bash
git lfs install
git lfs track "*.parquet"
git lfs track "data/**/*.h5"
git add .gitattributes
git commit -m "Setup Git LFS for large data files"
```

## Manual Cleanup

To manually clean up large files:

```bash
python scripts/cleanup_large_files.py --dry-run  # Preview actions
python scripts/cleanup_large_files.py --compress-logs  # Compress log files
python scripts/cleanup_large_files.py --move-large    # Move large files to archive
```

## Repository Health

Keep repository size reasonable:
- Target: < 100MB for fast cloning
- Large files in Git LFS or gitignored directories
- Regular cleanup of old logs and temporary files
