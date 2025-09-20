#!/usr/bin/env python3
"""Generate benchmark report from results."""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bench.benchcore.viz.report import generate_benchmark_report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate VIP benchmark report")
    parser.add_argument('--results-dir', '-r', required=True,
                       help='Path to benchmark results directory')
    parser.add_argument('--open', '-o', action='store_true',
                       help='Open the report after generation')
    
    args = parser.parse_args()
    
    # Check if results directory exists
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        sys.exit(1)
    
    print(f"📊 Generating report for: {results_dir}")
    
    try:
        # Generate report
        output_files = generate_benchmark_report(str(results_dir))
        
        print(f"✅ Report generated successfully!")
        print(f"📄 Markdown report: {output_files['report_md']}")
        print(f"📊 Plots directory: {output_files['figs_dir']}")
        print(f"🖼️  Generated {len(output_files['plots'])} plots")
        
        # Open report if requested
        if args.open:
            import webbrowser
            import os
            
            # Try to open the markdown file
            report_path = output_files['report_md']
            if os.path.exists(report_path):
                webbrowser.open(f'file://{os.path.abspath(report_path)}')
                print(f"🌐 Opened report in browser")
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
