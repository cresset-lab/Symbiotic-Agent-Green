from a2a.server.tasks import TaskUpdater
from messenger import Messenger


class Agent:
    def __init__(self):
        self.messenger = Messenger()
        # initialize other state here

    async def run(self, input_text: str, updater: TaskUpdater) -> None:
        """Implement your agent logic here.

        Args:
            input_text: The incoming message text
            updater: Report progress (update_status) and results (add_artifact)

        Use self.messenger.talk_to_agent(message, url) to call other agents.
        """
        raise NotImplementedError("Agent not implemented.")
