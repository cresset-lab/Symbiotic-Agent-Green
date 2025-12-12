from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Task,
    UnsupportedOperationError,
    InvalidRequestError,
)
from a2a.utils.errors import ServerError
from a2a.utils import (
    new_agent_text_message,
    new_task,
)

from agent import Agent


TERMINAL_STATES = ["completed", "canceled", "rejected", "failed"]


class Executor(AgentExecutor):
    def __init__(self):
        self.agent_store: dict[str, Agent] = {}

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        msg = context.message
        if not msg:
            raise ServerError(error=InvalidRequestError(message="Missing message in request"))

        task = context.current_task
        if not task or task.status.state in TERMINAL_STATES:
            task = new_task(msg)

        context_id = task.context_id
        agent = self.agent_store.get(context_id)
        if not agent:
            agent = Agent()
            self.agent_store[context_id] = agent

        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, context_id)
        await updater.start_work(new_agent_text_message(text="Thinking...", context_id=context_id, task_id=task.id))

        try:
            await agent.run(context.get_user_input(), updater)
            await updater.complete()
        except Exception as e:
            print(f"Task failed with agent error: {e}")
            await updater.failed(new_agent_text_message(f"Agent error: {e}", context_id=context_id))

    async def cancel(
        self, request: RequestContext, event_queue: EventQueue
    ) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())
