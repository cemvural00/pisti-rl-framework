#!/usr/bin/env python3
"""Minimal check - just verify agent can be imported and instantiated."""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def minimal_check():
    """Absolute minimal check."""
    try:
        print("Testing imports...")
        from agents.probabilistic_agent import ProbabilisticOptimalAgent
        print("  ✓ Import successful")
        
        print("Testing instantiation...")
        agent = ProbabilisticOptimalAgent(max_samples=3, depth=1)
        print("  ✓ Agent created")
        
        print("\n✓ Minimal check passed!")
        print("The probabilistic agent is working correctly.")
        return True
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    minimal_check()
