"""
Simple System Validation Script for Execution Feedback System
"""

import sys
sys.path.insert(0, '/home/bumblebee/Project/optimize')

from datetime import datetime
import sqlite3

print("\n" + "="*70)
print("EXECUTION FEEDBACK SYSTEM - VALIDATION TEST")
print("="*70)

# Test 1: Database tables
print("\n[1/5] Checking database tables...")
conn = sqlite3.connect('./data/lending_history.db')
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('virtual_orders', 'execution_statistics')")
tables = [row[0] for row in cursor.fetchall()]

if 'virtual_orders' in tables and 'execution_statistics' in tables:
    print("✅ Database tables exist")
else:
    print("❌ Missing tables")
    sys.exit(1)

# Test 2: Order creation
print("\n[2/5] Testing order creation...")
from ml_engine.order_manager import OrderManager

manager = OrderManager()
test_pred = {
    'currency': 'fUSD',
    'period': 30,
    'predicted_rate': 12.5,
    'confidence': 'High',
    'strategy': 'Test',
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

try:
    order_id = manager.create_virtual_order(test_pred)
    print(f"✅ Created order: {order_id[:8]}...")
except Exception as e:
    print(f"❌ Order creation failed: {e}")
    sys.exit(1)

# Test 3: Order retrieval
print("\n[3/5] Testing order retrieval...")
pending = manager.get_pending_orders()
print(f"✅ Retrieved {len(pending)} pending orders")

# Test 4: Execution features
print("\n[4/5] Testing execution features...")
from ml_engine.execution_features import ExecutionFeatures

calc = ExecutionFeatures()
try:
    features = calc.get_all_features('fUSD', 30)
    print(f"✅ Calculated {len(features)} features")
    print(f"   exec_rate_7d: {features['exec_rate_7d']:.2f}")
    print(f"   risk_adjustment_factor: {features['risk_adjustment_factor']:.2f}")
except Exception as e:
    print(f"❌ Feature calculation failed: {e}")
    sys.exit(1)

# Test 5: Execution stats
print("\n[5/5] Testing execution stats...")
try:
    stats = manager.get_execution_stats('fUSD', 30, 7)
    print(f"✅ Retrieved execution stats")
    print(f"   Total orders: {stats['total_orders']}")
    print(f"   Execution rate: {stats['execution_rate']:.1%}")
except Exception as e:
    print(f"❌ Stats query failed: {e}")
    sys.exit(1)

# Summary
print("\n" + "="*70)
print("✅ ALL VALIDATION TESTS PASSED")
print("="*70)

# Show current system state
cursor.execute("""
SELECT
    status,
    COUNT(*) as count
FROM virtual_orders
GROUP BY status
""")
status_counts = cursor.fetchall()

print("\nCurrent Virtual Orders Status:")
for status, count in status_counts:
    print(f"  {status}: {count}")

cursor.execute("SELECT COUNT(*) FROM funding_rates")
rate_count = cursor.fetchone()[0]
print(f"\nMarket Data: {rate_count:,} funding rate records")

conn.close()

print("\n🎉 Execution Feedback System is operational!")
print("\nNext steps:")
print("1. The system will create virtual orders after each prediction")
print("2. Orders are validated automatically when the validation window completes")
print("3. Execution history improves predictions over time")
print("4. Monitor execution rates via the /execution_stats API endpoint\n")
