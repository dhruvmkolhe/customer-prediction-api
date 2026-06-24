import subprocess
import sys
import os

def run_tests():
    """Run all tests using pytest."""
    print("=" * 60)
    print("  CUSTOMER PREDICTION SYSTEM - AUTOMATED TEST SUITE")
    print("=" * 60)
    
    # Ensure background server is NOT needed for these tests as they use TestClient
    # which mocks the server internally.
    
    # Run pytest
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "backend/tests", "-v"],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("  ✅ ALL TESTS PASSED SUCCESSFULLY!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("  ❌ SOME TESTS FAILED. PLEASE REVIEW LOGS ABOVE.")
            print("=" * 60)
            
        sys.exit(result.returncode)
        
    except Exception as e:
        print(f"\nError running tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
