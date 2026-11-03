"""Count local-model physical requests without applying cloud API prices."""


class RequestCountClient:
    def __init__(self, client, ledger, session):
        self.client, self.ledger, self.session = client, ledger, session

    def create_message(self, system, messages, tools, max_tokens=4096):
        rid = self.ledger.begin_request(self.session)
        try:
            result = self.client.create_message(system, messages, tools, max_tokens=max_tokens)
        except Exception:
            self.ledger.finish_request(rid, error=True)
            raise
        self.ledger.finish_request(rid, result.get('raw_usage', result.get('usage')))
        return result
