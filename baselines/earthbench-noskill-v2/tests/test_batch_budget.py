import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

from rsagent.billing.qwen import budget_for, ENDPOINT
from rsagent.evolution.request_ledger import RequestLedger, RequestBudgetExceeded


class SharedBudgetTests(unittest.TestCase):
    def test_pending_reservations_share_one_limit(self):
        config = {'provider':'openai','model':'qwen3.5-flash','base_url':ENDPOINT}
        with tempfile.TemporaryDirectory() as directory:
            ledger = RequestLedger(Path(directory), 100, budget=budget_for(config, '.001'))
            for i in range(16):
                ledger.register_session(str(i), max_requests=40)
            def reserve(i):
                try:
                    return ledger.begin_request(str(i),
                        reservation={'prompt_tokens':1000,'completion_tokens':100},
                        metadata={'model':'qwen3.5-flash','base_url':ENDPOINT,'cache_mode':None,
                                  'attempt':1,'reservation_basis':'offline_test'})
                except RequestBudgetExceeded:
                    return None
            with ThreadPoolExecutor(max_workers=16) as pool:
                accepted = [r for r in pool.map(reserve,range(16)) if r]
            self.assertEqual(len(accepted), 2)
            self.assertLessEqual(Decimal(ledger.summary()['budget']['estimated_committed_cost']),Decimal('.001'))
            for r in accepted:
                ledger.finish_request(r, {'prompt_tokens':1000,'completion_tokens':100,'total_tokens':1100})
            self.assertEqual(ledger.summary()['requests_pending'],0)


if __name__ == '__main__':
    unittest.main()
