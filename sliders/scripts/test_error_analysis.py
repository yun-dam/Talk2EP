#!/usr/bin/env python3
"""Test script for the Error Analysis Viewer"""

import os
import sys

from sliders.utils import logger

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from .error_analysis_viewer import ErrorAnalysisViewer


def test_basic_functionality():
    """Test basic functionality of the Error Analysis Viewer"""
    print("🧪 Testing Error Analysis Viewer...")

    # Initialize viewer
    viewer = ErrorAnalysisViewer()

    # Test scanning experiments
    print("📁 Scanning for experiments...")
    experiments = viewer.scan_experiments()
    print(f"Found {len(experiments)} experiments")

    if experiments:
        print("✅ Experiment scanning works!")
        print("Sample experiments:")
        for i, exp in enumerate(experiments[:3]):  # Show first 3
            print(f"  {i}: {exp}")
    else:
        print("⚠️  No experiments found - this might be normal if no matching files exist")

    # Test question extraction (if we have experiments)
    if viewer.experiments:
        print("\n🔍 Testing question extraction...")
        try:
            # Try to load the first experiment
            result = viewer.load_experiment(0)
            print(f"Load result: {result}")

            if viewer.current_questions:
                print(f"✅ Successfully extracted {len(viewer.current_questions)} questions")

                # Test question summary
                if viewer.current_questions:
                    summary = viewer.get_question_summary(0)
                    logger.info(f"Question summary: {summary}")
                    print("✅ Question summary generation works!")

                    # Test detail extraction
                    details = viewer.get_question_details(0, "schema")
                    logger.info(f"Question details: {details}")
                    print("✅ Detail extraction works!")

                    # Test statistics
                    stats = viewer.get_experiment_statistics()
                    logger.info(f"Experiment statistics: {stats}")
                    print("✅ Statistics generation works!")

            else:
                print("⚠️  No questions extracted - this might be normal for some log formats")

        except Exception as e:
            print(f"❌ Error during testing: {e}")
            import traceback

            traceback.print_exc()

    print("\n🎉 Basic functionality test completed!")


if __name__ == "__main__":
    test_basic_functionality()
