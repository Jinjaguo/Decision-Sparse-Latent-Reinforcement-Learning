from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from decision_sparse_rl.metrics.exp18 import selector_metrics


def test_selector_metrics_counts_recovery_and_headroom():
    default=[{"success":True,"safety_stop":False},{"success":False,"safety_stop":False}]
    selected=[{"success":True,"safety_stop":False},{"success":True,"safety_stop":False}]
    oracle=selected
    result=selector_metrics(selected,default,oracle)
    assert result["safe_success_rate"]==1 and result["demand_recovery_rate"]==1 and result["oracle_headroom_capture"]==1

