#!/usr/bin/env python3
"""
Run All PCB Engine Unit Tests
=============================

Runs all unit tests for the PCB Engine pistons and generates a coverage report.

Usage:
    python run_all_tests.py
    python run_all_tests.py -v  # Verbose
    python run_all_tests.py --coverage  # With coverage report
"""

import sys
import os
import unittest
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def run_tests(verbosity=2, pattern='test_*.py'):
    """Run all tests and return results."""
    # Get the tests directory
    tests_dir = os.path.dirname(os.path.abspath(__file__))

    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.discover(tests_dir, pattern=pattern)

    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    return result


def print_summary(result):
    """Print test summary."""
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    passed = total - failures - errors - skipped

    print(f"Total:    {total}")
    print(f"Passed:   {passed}")
    print(f"Failed:   {failures}")
    print(f"Errors:   {errors}")
    print(f"Skipped:  {skipped}")
    print()

    if result.wasSuccessful():
        print("SUCCESS - All tests passed!")
    else:
        print("FAILED - Some tests did not pass")

        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}")

        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}")

    print("=" * 70)
    return result.wasSuccessful()


def main():
    parser = argparse.ArgumentParser(description='Run PCB Engine unit tests')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Minimal output')
    parser.add_argument('--pattern', default='test_*.py',
                        help='Test file pattern (default: test_*.py)')
    parser.add_argument('--coverage', action='store_true',
                        help='Run with coverage report')

    args = parser.parse_args()

    verbosity = 2
    if args.verbose:
        verbosity = 3
    elif args.quiet:
        verbosity = 1

    print("\n" + "=" * 70)
    print("PCB ENGINE UNIT TEST SUITE")
    print("=" * 70)
    print()

    if args.coverage:
        try:
            import coverage
            cov = coverage.Coverage(source=['pcb_engine'])
            cov.start()
            result = run_tests(verbosity, args.pattern)
            cov.stop()
            cov.save()
            print("\nCoverage Report:")
            cov.report()
        except ImportError:
            print("Coverage not installed. Run: pip install coverage")
            result = run_tests(verbosity, args.pattern)
    else:
        result = run_tests(verbosity, args.pattern)

    success = print_summary(result)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
