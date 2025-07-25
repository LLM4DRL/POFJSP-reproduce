#!/usr/bin/env python3
"""
Large File Cleanup Script

Manages large files in the repository to improve clone times and reduce storage.
Implements proper data management practices for POFJSP project.
"""

import os
import shutil
import argparse
import logging
from pathlib import Path
from typing import List, Tuple
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_file_size_mb(file_path: Path) -> float:
    """Get file size in MB."""
    return file_path.stat().st_size / (1024 * 1024)


def find_large_files(directory: Path, min_size_mb: float = 10.0) -> List[Tuple[Path, float]]:
    """Find files larger than specified size."""
    large_files = []
    
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            try:
                size_mb = get_file_size_mb(file_path)
                if size_mb > min_size_mb:
                    large_files.append((file_path, size_mb))
            except (OSError, PermissionError):
                continue
    
    return sorted(large_files, key=lambda x: x[1], reverse=True)


def compress_log_files(log_files: List[Path]) -> None:
    """Compress large log files to save space."""
    import gzip
    
    for log_file in log_files:
        if log_file.suffix == '.log':
            compressed_file = log_file.with_suffix('.log.gz')
            
            logger.info(f"Compressing {log_file} -> {compressed_file}")
            
            try:
                with open(log_file, 'rb') as f_in:
                    with gzip.open(compressed_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Remove original if compression successful
                original_size = get_file_size_mb(log_file)
                compressed_size = get_file_size_mb(compressed_file)
                compression_ratio = compressed_size / original_size
                
                logger.info(f"Compression: {original_size:.1f}MB -> {compressed_size:.1f}MB "
                           f"({compression_ratio:.1%} of original)")
                
                log_file.unlink()  # Remove original
                
            except Exception as e:
                logger.error(f"Failed to compress {log_file}: {e}")
                if compressed_file.exists():
                    compressed_file.unlink()


def move_to_gitignore_location(file_path: Path, repo_root: Path) -> None:
    """Move large files to gitignored location."""
    # Create archive directory structure
    archive_dir = repo_root / "archive" / "large_files"
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # Preserve relative structure
    relative_path = file_path.relative_to(repo_root)
    destination = archive_dir / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Moving {file_path} -> {destination}")
    shutil.move(str(file_path), str(destination))


def setup_git_lfs_for_data_files(repo_root: Path) -> None:
    """Set up Git LFS for large data files."""
    gitattributes_file = repo_root / ".gitattributes"
    
    lfs_patterns = [
        "*.parquet filter=lfs diff=lfs merge=lfs -text",
        "*.h5 filter=lfs diff=lfs merge=lfs -text", 
        "*.hdf5 filter=lfs diff=lfs merge=lfs -text",
        "*.pkl filter=lfs diff=lfs merge=lfs -text",
        "*.npy filter=lfs diff=lfs merge=lfs -text",
        "*.npz filter=lfs diff=lfs merge=lfs -text",
        "data/**/*.parquet filter=lfs diff=lfs merge=lfs -text"
    ]
    
    # Read existing patterns
    existing_patterns = set()
    if gitattributes_file.exists():
        with open(gitattributes_file, 'r') as f:
            existing_patterns = set(line.strip() for line in f if line.strip())
    
    # Add new patterns
    new_patterns = []
    for pattern in lfs_patterns:
        if pattern not in existing_patterns:
            new_patterns.append(pattern)
    
    if new_patterns:
        logger.info(f"Adding {len(new_patterns)} Git LFS patterns to .gitattributes")
        with open(gitattributes_file, 'a') as f:
            f.write("\n# Large data files for Git LFS\n")
            for pattern in new_patterns:
                f.write(f"{pattern}\n")


def create_data_management_readme(repo_root: Path) -> None:
    """Create README for data management."""
    readme_content = """# Data Management

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
"""
    
    readme_file = repo_root / "docs" / "data_management.md"
    readme_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(readme_file, 'w') as f:
        f.write(readme_content)
    
    logger.info(f"Created data management documentation: {readme_file}")


def main():
    """Main cleanup function."""
    parser = argparse.ArgumentParser(description='Clean up large files in POFJSP repository')
    parser.add_argument('--dry-run', action='store_true', help='Preview actions without executing')
    parser.add_argument('--compress-logs', action='store_true', help='Compress large log files')
    parser.add_argument('--move-large', action='store_true', help='Move large files to archive')
    parser.add_argument('--setup-lfs', action='store_true', help='Setup Git LFS for data files')
    parser.add_argument('--min-size', type=float, default=10.0, help='Minimum file size in MB')
    
    args = parser.parse_args()
    
    # Find repository root
    repo_root = Path(__file__).parent.parent
    
    logger.info(f"Scanning repository: {repo_root}")
    
    # Find large files
    large_files = find_large_files(repo_root, args.min_size)
    
    if not large_files:
        logger.info("No large files found!")
        return
    
    logger.info(f"Found {len(large_files)} files larger than {args.min_size}MB:")
    total_size = 0
    for file_path, size_mb in large_files:
        logger.info(f"  {file_path.relative_to(repo_root)}: {size_mb:.1f}MB")
        total_size += size_mb
    
    logger.info(f"Total size: {total_size:.1f}MB")
    
    if args.dry_run:
        logger.info("DRY RUN - No changes made")
        return
    
    # Process files based on type
    log_files = []
    data_files = []
    other_files = []
    
    for file_path, size_mb in large_files:
        if file_path.suffix in ['.log']:
            log_files.append(file_path)
        elif file_path.suffix in ['.parquet', '.h5', '.hdf5', '.pkl', '.npy', '.npz']:
            data_files.append(file_path)
        else:
            other_files.append(file_path)
    
    # Compress log files
    if args.compress_logs and log_files:
        logger.info(f"Compressing {len(log_files)} log files...")
        compress_log_files(log_files)
    
    # Move large files to archive
    if args.move_large:
        files_to_move = data_files + other_files
        if files_to_move:
            logger.info(f"Moving {len(files_to_move)} large files to archive...")
            for file_path in files_to_move:
                move_to_gitignore_location(file_path, repo_root)
    
    # Setup Git LFS
    if args.setup_lfs:
        logger.info("Setting up Git LFS for data files...")
        setup_git_lfs_for_data_files(repo_root)
    
    # Create documentation
    create_data_management_readme(repo_root)
    
    logger.info("Cleanup completed!")


if __name__ == "__main__":
    main()