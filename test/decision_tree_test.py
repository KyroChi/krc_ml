import numpy as np
import krc_ml_lib.decision_tree as dt

np.random.seed(42)

PASSED_TESTS = 0
FAILED_TESTS = 0

# ===========================================================================
# Test 1: Single Gaussian distribution with a single label
# ===========================================================================
def test1(passed, failed):
    num_points = 100
    data = np.array([[np.random.normal(0, 1), np.random.normal(0, 1)] for _ in range(num_points)])
    y = np.array([1 for _ in range(num_points)])

    tree = dt.MSECARTTree()
    tree.fit(data, y)

    if np.any(y - tree.predict(data) != 0):
        failed += 1
    else:
        passed += 1

    return passed, failed

# ===========================================================================
# Test 2: Two Distinct Gaussian Distributions
# ===========================================================================
def test2(passed, failed):
    num_points = 100
    data1 = np.array([[np.random.normal(-1, .1), np.random.normal(0, .1)] for _ in range(num_points)])
    data2 = np.array([[np.random.normal(1, .1), np.random.normal(0, .1)] for _ in range(num_points)])
    y1 = np.array([0 for _ in range(num_points)])
    y2 = np.array([1 for _ in range(num_points)])

    data = np.append(data1, data2, axis=0)
    y = np.append(y1, y2, axis=0)

    tree = dt.MSECARTTree()
    tree.fit(data, y)

    test1 = np.array([[np.random.normal(-1, .1), np.random.normal(0, .1)] for _ in range(num_points)])
    test2 = np.array([[np.random.normal(1, .1), np.random.normal(0, .1)] for _ in range(num_points)])

    if np.any(y1 - tree.predict(test1) != 0) or np.any(y2 - tree.predict(test2) != 0):
        failed += 1
    else:
        passed += 1

    return passed, failed

# # Also check that we can split along a different feature
def test3(passed, failed):
    num_points = 100
    data1 = np.array([[np.random.normal(0, .1), np.random.normal(-1, .1)] for _ in range(num_points)])
    data2 = np.array([[np.random.normal(0, .1), np.random.normal(1, .1)] for _ in range(num_points)])
    y1 = np.array([0 for _ in range(num_points)])
    y2 = np.array([1 for _ in range(num_points)])

    data = np.append(data1, data2, axis=0)
    y = np.append(y1, y2, axis=0)

    tree = dt.MSECARTTree()
    tree.fit(data, y)

    test1 = np.array([[np.random.normal(0, .1), np.random.normal(-1, .1)] for _ in range(num_points)])
    test2 = np.array([[np.random.normal(0, .1), np.random.normal(1, .1)] for _ in range(num_points)])

    if np.any(y1 - tree.predict(test1) != 0) or np.any(y2 - tree.predict(test2) != 0):
        failed += 1
    else:
        passed += 1

    return passed, failed

# ===========================================================================
# Test 3: XOR
# ===========================================================================
def test4(passed, failed):
    num_points = 4
    std_dev = 0.15

    data1 = np.array([[np.random.normal(1, std_dev), np.random.normal(1, std_dev)] for _ in range(num_points)])
    data2 = np.array([[np.random.normal(1, std_dev), np.random.normal(-1, std_dev)] for _ in range(num_points)])
    data3 = np.array([[np.random.normal(-1, std_dev), np.random.normal(-1, std_dev)] for _ in range(num_points)])
    data4 = np.array([[np.random.normal(-1, std_dev), np.random.normal(1, std_dev)] for _ in range(num_points)])

    y1 = np.array([0 for _ in range(num_points)])
    y2 = np.array([1 for _ in range(num_points)])
    y3 = np.array([0 for _ in range(num_points)])
    y4 = np.array([1 for _ in range(num_points)])

    data = data1
    y = y1

    for d in [data2, data3, data4]:
        data = np.append(data, d, axis=0)

    for yy in [y2, y3, y4]:
        y = np.append(y, yy, axis=0)

    tree = dt.MSECARTTree(max_depth=3)
    tree.fit(data, y)

    data1 = np.array([[np.random.normal(1, std_dev), np.random.normal(1, std_dev)] for _ in range(num_points)])
    data2 = np.array([[np.random.normal(1, std_dev), np.random.normal(-1, std_dev)] for _ in range(num_points)])
    data3 = np.array([[np.random.normal(-1, std_dev), np.random.normal(-1, std_dev)] for _ in range(num_points)])
    data4 = np.array([[np.random.normal(-1, std_dev), np.random.normal(1, std_dev)] for _ in range(num_points)])

    y1 = np.array([0 for _ in range(num_points)])
    y2 = np.array([1 for _ in range(num_points)])
    y3 = np.array([0 for _ in range(num_points)])
    y4 = np.array([1 for _ in range(num_points)])

    for pp in [[data1, y1], [data2, y2], [data3, y3], [data4, y4]]:
        X = pp[0]
        y = pp[1]

        yhat = tree.predict(X)
        if np.any(y - yhat != 0):
            failed += 1
        else:
            passed += 1

    return passed, failed


# ===========================================================================
# Final printout
# ===========================================================================
# PASSED_TESTS, FAILED_TESTS = test1(PASSED_TESTS, FAILED_TESTS)
# PASSED_TESTS, FAILED_TESTS = test2(PASSED_TESTS, FAILED_TESTS)
# PASSED_TESTS, FAILED_TESTS = test3(PASSED_TESTS, FAILED_TESTS)
PASSED_TESTS, FAILED_TESTS = test4(PASSED_TESTS, FAILED_TESTS)

print("Failed: %d\nPassed: %d\nTotal: %d" % (FAILED_TESTS, PASSED_TESTS, FAILED_TESTS + PASSED_TESTS))