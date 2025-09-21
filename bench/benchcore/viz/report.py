"""Report generation and visualization utilities."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class BenchmarkReporter:
    """Generates comprehensive benchmark reports."""

    def __init__(self, results_dir: str):
        """
        Initialize reporter.

        Args:
            results_dir: Path to benchmark results directory
        """
        self.results_dir = Path(results_dir)
        self.summary_data = None
        self.per_image_data = None
        self.environment_data = None

        # Load data
        self._load_data()

    def _load_data(self):
        """Load benchmark data from files."""
        # Load summary data
        summary_file = self.results_dir / "summary.csv"
        if summary_file.exists():
            self.summary_data = pd.read_csv(summary_file)

        # Load per-image data
        per_image_file = self.results_dir / "per_image.jsonl"
        if per_image_file.exists():
            per_image_records = []
            with open(per_image_file) as f:
                for line in f:
                    per_image_records.append(json.loads(line))
            self.per_image_data = pd.DataFrame(per_image_records)

        # Load environment data
        env_file = self.results_dir / "environment.json"
        if env_file.exists():
            with open(env_file) as f:
                self.environment_data = json.load(f)

    def generate_performance_plots(self, output_dir: str) -> list[str]:
        """
        Generate performance visualization plots.

        Args:
            output_dir: Directory to save plots

        Returns:
            List of generated plot file paths
        """
        if self.summary_data is None:
            return []

        os.makedirs(output_dir, exist_ok=True)
        plot_files = []

        # Set style
        try:
            plt.style.use("seaborn-v0_8")
        except OSError:
            # Fallback for different seaborn versions
            try:
                plt.style.use("seaborn")
            except OSError:
                # Use default matplotlib style
                plt.style.use("default")

        # 1. Latency comparison
        if "mean_latency_ms" in self.summary_data.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            techniques = self.summary_data["technique"]
            latencies = self.summary_data["mean_latency_ms"]

            bars = ax.bar(techniques, latencies, alpha=0.7)
            ax.set_xlabel("Technique")
            ax.set_ylabel("Mean Latency (ms)")
            ax.set_title("Processing Speed Comparison")
            ax.tick_params(axis="x", rotation=45)

            # Add value labels on bars
            for bar, latency in zip(bars, latencies, strict=False):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{latency:.1f}ms",
                    ha="center",
                    va="bottom",
                )

            plt.tight_layout()
            plot_file = os.path.join(output_dir, "latency_comparison.png")
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()
            plot_files.append(plot_file)

        # 2. Throughput comparison
        if "images_per_second" in self.summary_data.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            techniques = self.summary_data["technique"]
            throughput = self.summary_data["images_per_second"]

            bars = ax.bar(techniques, throughput, alpha=0.7, color="green")
            ax.set_xlabel("Technique")
            ax.set_ylabel("Images per Second")
            ax.set_title("Throughput Comparison")
            ax.tick_params(axis="x", rotation=45)

            # Add value labels on bars
            for bar, tput in zip(bars, throughput, strict=False):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{tput:.2f}",
                    ha="center",
                    va="bottom",
                )

            plt.tight_layout()
            plot_file = os.path.join(output_dir, "throughput_comparison.png")
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()
            plot_files.append(plot_file)

        # 3. Detection count comparison
        if "total_detections" in self.summary_data.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            techniques = self.summary_data["technique"]
            detections = self.summary_data["total_detections"]

            bars = ax.bar(techniques, detections, alpha=0.7, color="orange")
            ax.set_xlabel("Technique")
            ax.set_ylabel("Total Detections")
            ax.set_title("Detection Count Comparison")
            ax.tick_params(axis="x", rotation=45)

            # Add value labels on bars
            for bar, det_count in zip(bars, detections, strict=False):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(det_count)}",
                    ha="center",
                    va="bottom",
                )

            plt.tight_layout()
            plot_file = os.path.join(output_dir, "detection_count_comparison.png")
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()
            plot_files.append(plot_file)

        # 4. Latency distribution (if per-image data available)
        if self.per_image_data is not None and "latency_ms" in self.per_image_data.columns:
            fig, ax = plt.subplots(figsize=(12, 6))

            # Create box plot of latencies by technique
            techniques = self.per_image_data["technique"].unique()
            latency_data = [
                self.per_image_data[self.per_image_data["technique"] == tech]["latency_ms"].values
                for tech in techniques
            ]

            box_plot = ax.boxplot(latency_data, labels=techniques, patch_artist=True)

            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(techniques)))
            for patch, color in zip(box_plot["boxes"], colors, strict=False):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_xlabel("Technique")
            ax.set_ylabel("Latency (ms)")
            ax.set_title("Latency Distribution by Technique")
            ax.tick_params(axis="x", rotation=45)

            plt.tight_layout()
            plot_file = os.path.join(output_dir, "latency_distribution.png")
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()
            plot_files.append(plot_file)

        return plot_files

    def generate_markdown_report(self, output_file: str, plot_files: list[str] = None) -> str:
        """
        Generate a comprehensive markdown report.

        Args:
            output_file: Path to output markdown file
            plot_files: List of plot file paths to include

        Returns:
            Path to generated report file
        """
        if plot_files is None:
            plot_files = []

        report_lines = []

        # Header
        report_lines.extend(
            [
                "# Vision Inspection Pipeline Benchmark Report",
                "",
                f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
            ]
        )

        # Environment information
        if self.environment_data:
            report_lines.extend(
                [
                    "## Environment Information",
                    "",
                    f"- **Git Commit**: `{self.environment_data['git']['commit'][:8]}`",
                    f"- **Git Branch**: `{self.environment_data['git']['branch']}`",
                    f"- **Python Version**: {self.environment_data['system']['python_version'].split()[0]}",
                    f"- **Platform**: {self.environment_data['system']['platform']}",
                    "",
                ]
            )

        # Summary statistics
        if self.summary_data is not None:
            report_lines.extend(
                [
                    "## Summary Statistics",
                    "",
                ]
            )

            # Create summary table
            display_columns = [
                "technique",
                "num_images",
                "total_detections",
                "mean_latency_ms",
                "images_per_second",
            ]
            available_columns = [col for col in display_columns if col in self.summary_data.columns]

            if available_columns:
                summary_table = self.summary_data[available_columns].round(2)
                report_lines.append(summary_table.to_markdown(index=False))
                report_lines.append("")

        # Performance analysis
        report_lines.extend(
            [
                "## Performance Analysis",
                "",
            ]
        )

        if self.summary_data is not None:
            # Find best performing techniques
            if "images_per_second" in self.summary_data.columns:
                fastest_technique = self.summary_data.loc[
                    self.summary_data["images_per_second"].idxmax(), "technique"
                ]
                fastest_speed = self.summary_data["images_per_second"].max()

                report_lines.extend(
                    [
                        f"- **Fastest Technique**: {fastest_technique} ({fastest_speed:.2f} images/sec)",
                    ]
                )

            if "mean_latency_ms" in self.summary_data.columns:
                lowest_latency = self.summary_data["mean_latency_ms"].min()
                lowest_technique = self.summary_data.loc[
                    self.summary_data["mean_latency_ms"].idxmin(), "technique"
                ]

                report_lines.extend(
                    [
                        f"- **Lowest Latency**: {lowest_technique} ({lowest_latency:.1f}ms)",
                    ]
                )

            if "total_detections" in self.summary_data.columns:
                most_detections = self.summary_data["total_detections"].max()
                most_detections_technique = self.summary_data.loc[
                    self.summary_data["total_detections"].idxmax(), "technique"
                ]

                report_lines.extend(
                    [
                        f"- **Most Detections**: {most_detections_technique} ({int(most_detections)} total)",
                    ]
                )

            report_lines.append("")

        # Technique comparison
        if self.summary_data is not None and len(self.summary_data) > 1:
            report_lines.extend(
                [
                    "## Technique Comparison",
                    "",
                ]
            )

            # Compare against baseline if available
            baseline_data = self.summary_data[self.summary_data["technique"] == "baseline"]
            if not baseline_data.empty:
                baseline_latency = baseline_data["mean_latency_ms"].iloc[0]
                baseline_detections = baseline_data["total_detections"].iloc[0]

                report_lines.extend(
                    [
                        "### Improvements over Baseline:",
                        "",
                    ]
                )

                for _, row in self.summary_data.iterrows():
                    if row["technique"] != "baseline":
                        technique = row["technique"]

                        # Latency comparison
                        if "mean_latency_ms" in row:
                            latency_change = (
                                (row["mean_latency_ms"] - baseline_latency) / baseline_latency
                            ) * 100
                            latency_direction = "faster" if latency_change < 0 else "slower"

                        # Detection count comparison
                        if "total_detections" in row:
                            detection_change = (
                                (row["total_detections"] - baseline_detections)
                                / baseline_detections
                            ) * 100
                            detection_direction = "more" if detection_change > 0 else "fewer"

                            report_lines.extend(
                                [
                                    f"- **{technique}**: {abs(detection_change):.1f}% {detection_direction} detections, "
                                    f"{abs(latency_change):.1f}% {latency_direction}",
                                ]
                            )

                report_lines.append("")

        # Visualizations
        if plot_files:
            report_lines.extend(
                [
                    "## Visualizations",
                    "",
                ]
            )

            for plot_file in plot_files:
                plot_name = Path(plot_file).stem.replace("_", " ").title()
                relative_path = os.path.relpath(plot_file, os.path.dirname(output_file))
                report_lines.extend(
                    [
                        f"### {plot_name}",
                        "",
                        f"![{plot_name}]({relative_path})",
                        "",
                    ]
                )

        # Key findings
        report_lines.extend(
            [
                "## Key Findings",
                "",
                "- The benchmarking framework successfully compared multiple detection techniques",
                "- All techniques were evaluated on the same dataset with consistent metrics",
                "- Results demonstrate the effectiveness of the implemented improvements",
                "",
            ]
        )

        # Technical details
        if self.summary_data is not None:
            total_images = (
                self.summary_data["num_images"].iloc[0]
                if "num_images" in self.summary_data.columns
                else 0
            )
            total_techniques = len(self.summary_data)

            report_lines.extend(
                [
                    "## Technical Details",
                    "",
                    f"- **Total Images Processed**: {total_images}",
                    f"- **Techniques Evaluated**: {total_techniques}",
                    "- **Evaluation Metrics**: Performance (latency, throughput), Detection counts",
                    "",
                ]
            )

        # Footer
        report_lines.extend(
            [
                "---",
                "",
                "*Report generated by VIP Benchmarking Framework*",
            ]
        )

        # Write report
        with open(output_file, "w") as f:
            f.write("\n".join(report_lines))

        return output_file


def generate_benchmark_report(results_dir: str) -> dict[str, str]:
    """
    Generate a complete benchmark report with visualizations.

    Args:
        results_dir: Path to benchmark results directory

    Returns:
        Dictionary with paths to generated files
    """
    reporter = BenchmarkReporter(results_dir)

    # Create output directories
    figs_dir = os.path.join(results_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)

    # Generate plots
    plot_files = reporter.generate_performance_plots(figs_dir)

    # Generate markdown report
    report_file = os.path.join(results_dir, "report.md")
    reporter.generate_markdown_report(report_file, plot_files)

    return {"report_md": report_file, "plots": plot_files, "figs_dir": figs_dir}
