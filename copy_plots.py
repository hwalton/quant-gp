#!/usr/bin/env python3
"""
Copy existing plots to web directory.
"""
import os
import shutil
from pathlib import Path

def copy_plots_to_web():
    """Copy any existing plots to the web static directory."""
    web_dir = Path("/home/harvey/Git/quant-gp/web/static/images")
    web_dir.mkdir(parents=True, exist_ok=True)
    
    # Sources to check for plots
    sources = [
        Path("/home/harvey/Git/quant-gp/gp-model/outputs"),
        Path("/home/harvey/Git/quant-gp/3-optimise_portfolio"),
        Path("/home/harvey/Git/quant-gp")  # Root directory
    ]
    
    copied_files = []
    
    for source_dir in sources:
        if source_dir.exists():
            for plot_file in source_dir.glob("*.png"):
                dest_file = web_dir / plot_file.name
                shutil.copy2(plot_file, dest_file)
                copied_files.append(f"{plot_file} -> {dest_file}")
                print(f"📂 Copied: {plot_file.name}")
    
    if copied_files:
        print(f"\n✅ Copied {len(copied_files)} plot files to web/static/images/")
        print("\nCopied files:")
        for file_info in copied_files:
            print(f"  • {file_info}")
    else:
        print("❌ No plot files found to copy")
    
    # List what's now in the web directory
    if web_dir.exists():
        web_files = list(web_dir.glob("*.png"))
        print(f"\n📁 Web static images directory now contains:")
        for file in web_files:
            file_size = file.stat().st_size / 1024  # KB
            print(f"  • {file.name} ({file_size:.1f} KB)")

if __name__ == "__main__":
    copy_plots_to_web()
