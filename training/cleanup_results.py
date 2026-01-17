"""Utility script to clean up evaluation results."""

import argparse
import os
import shutil
from pathlib import Path
from typing import List, Optional


def cleanup_results(
    results_dir: str = "results",
    keep_recent: Optional[int] = None,
    pattern: Optional[str] = None,
    dry_run: bool = False,
) -> List[str]:
    """
    Clean up old result directories.
    
    Args:
        results_dir: Base results directory
        keep_recent: Keep only N most recent results (by modification time)
        pattern: Only delete directories matching pattern (e.g., "eval_2024*")
        dry_run: If True, only print what would be deleted
    
    Returns:
        List of deleted directory paths
    """
    if not os.path.exists(results_dir):
        print(f"Results directory does not exist: {results_dir}")
        return []
    
    # Get all subdirectories
    all_dirs = [
        os.path.join(results_dir, d)
        for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ]
    
    # Filter by pattern if provided
    if pattern:
        import fnmatch
        all_dirs = [d for d in all_dirs if fnmatch.fnmatch(os.path.basename(d), pattern)]
    
    if not all_dirs:
        print("No directories found matching criteria.")
        return []
    
    # Sort by modification time (most recent first)
    all_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Determine which to delete
    if keep_recent is not None and keep_recent > 0:
        dirs_to_delete = all_dirs[keep_recent:]
        dirs_to_keep = all_dirs[:keep_recent]
    else:
        dirs_to_delete = all_dirs
        dirs_to_keep = []
    
    if not dirs_to_delete:
        print("No directories to delete.")
        return []
    
    # Print what will be deleted
    print(f"\nFound {len(all_dirs)} directories")
    if dirs_to_keep:
        print(f"Keeping {len(dirs_to_keep)} most recent:")
        for d in dirs_to_keep:
            print(f"  - {d}")
    print(f"\n{'Would delete' if dry_run else 'Deleting'} {len(dirs_to_delete)} directories:")
    for d in dirs_to_delete:
        print(f"  - {d}")
    
    if dry_run:
        print("\n(Dry run - no files were actually deleted)")
        return []
    
    # Delete directories
    deleted = []
    for d in dirs_to_delete:
        try:
            shutil.rmtree(d)
            deleted.append(d)
            print(f"Deleted: {d}")
        except Exception as e:
            print(f"Error deleting {d}: {e}")
    
    print(f"\nDeleted {len(deleted)} directories.")
    return deleted


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Clean up old evaluation result directories"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Base results directory (default: results)",
    )
    parser.add_argument(
        "--keep-recent",
        type=int,
        default=None,
        help="Keep only N most recent results (default: delete all matching)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Only delete directories matching pattern (e.g., 'eval_2024*')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    
    args = parser.parse_args()
    
    cleanup_results(
        results_dir=args.results_dir,
        keep_recent=args.keep_recent,
        pattern=args.pattern,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
