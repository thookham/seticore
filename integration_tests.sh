#!/bin/bash -e

# Runs the integration tests. These require a bunch of specific local data but
# they catch more bugs than just the unit tests.

echo integration testing...

if [ -f "./build/beamforming_integration_test" ]; then
    ./build/beamforming_integration_test
else
    echo "Skipping beamforming_integration_test (binary not found)"
fi

if [ -f "./test_extract.sh" ] && [ -f "./build/extract" ]; then
    ./test_extract.sh
else
    echo "Skipping test_extract.sh (binary or script not found)"
fi
