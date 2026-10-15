# [MODULE DOC] Progress-bar shim. If env var M4_DISABLE_RICH == "1" or the `rich` package is missing, sets the
# module
# flag M4_DISABLE_RICH and defines minimal plain-text stand-ins for rich.progress classes (BarColumn,
# MofNCompleteColumn, TaskProgressColumn, TimeElapsedColumn, Progress) plus a Task record; otherwise it
# re-exports those classes from rich.progress. Stand-in columns expose render(task) returning a string.
import os
import time


M4_DISABLE_RICH = False
if os.environ.get("M4_DISABLE_RICH", "") == "1":
    M4_DISABLE_RICH = True
else:
    try:
        import rich  # noqa
    except ModuleNotFoundError:
        M4_DISABLE_RICH = True

if not M4_DISABLE_RICH:
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TimeElapsedColumn

else:
    # This is a simple equivalent of some of the `rich`'s classes we use but which doesn't use
    # `rich` or any formatting. We use it if there is no `rich` installed or during HPC training
    # where we don't use a live console and log to a file instead - so we want easy to read logs and
    # `rich`'s output mangling causes more trouble than it helps.

    class BarColumn:
        # [CLASS DOC] Plain stand-in for rich's BarColumn; draws nothing.
        def render(self, task):
            # [FUNCTION DOC] Returns an empty string regardless of task.
            return ""

    class MofNCompleteColumn:
        # [CLASS DOC] Plain stand-in for rich's MofNCompleteColumn ("completed/total" text).
        def render(self, task):
            # [FUNCTION DOC] Returns "<task.completed>/<task.total_steps>", using "UNK" when total_steps is
            # None.
            if task.total_steps is not None:
                total_steps = task.total_steps
            else:
                total_steps = "UNK"
            return f"{task.completed}/{total_steps}"

    class TaskProgressColumn:
        # [CLASS DOC] Plain stand-in for rich's TaskProgressColumn (percentage text).
        def render(self, task):
            # [FUNCTION DOC] Returns integer percentage completed/total_steps*100 right-aligned to width 3
            # followed by "%", or "UNK%"
            # when total_steps is None. Raises ZeroDivisionError if total_steps is 0.
            if task.total_steps is not None:
                percent = int(task.completed / task.total_steps * 100)
                return f"{percent:>3}%"
            else:
                return "UNK%"

    class TimeElapsedColumn:
        # [CLASS DOC] Plain stand-in for rich's TimeElapsedColumn.
        def render(self, task):
            # [FUNCTION DOC] Returns elapsed time since task.start_time formatted "D:HH:MM:SS" (days derived
            # from day-of-year minus 1,
            # so only correct for elapsed times under one year).
            time_diff = time.gmtime(time.time() - task.start_time)
            days = int(time.strftime("%j", time_diff)) - 1
            time_str = time.strftime("%H:%M:%S", time_diff)
            return f"{days}:{time_str}"

    class Task:
        # [CLASS DOC] Simple progress record: description, total_steps, completed counter and start_time.
        def __init__(self, description, total_steps, *args, **kwargs):
            # [FUNCTION DOC] Stores description and total_steps (may be None), sets completed to 0 and
            # start_time to now. Extra
            # args/kwargs are accepted and ignored for API compatibility.
            self.description = description
            self.total_steps = total_steps

            self.completed = 0
            self.start_time = time.time()

        def step(self, advance_steps):
            # [FUNCTION DOC] Adds advance_steps to the completed counter.
            self.completed += advance_steps

    class Progress:
        # [CLASS DOC] Minimal stand-in for rich.progress.Progress: a context manager holding a list of Task
        # objects; renders
        # nothing by itself.
        def __init__(self, *args, **kwargs):
            # [FUNCTION DOC] Initializes an empty task list and description "Progress"; all arguments (column
            # objects etc.) are ignored.
            self.tasks = []
            self.description = "Progress"

        def __enter__(self):
            # [FUNCTION DOC] Context-manager entry; returns self.
            return self

        def __exit__(self, *args):
            # [FUNCTION DOC] Context-manager exit; does nothing and does not suppress exceptions.
            pass

        def update(self, task, advance):
            # [FUNCTION DOC] Advances the given Task by `advance` steps and returns self. Unlike rich, `task`
            # is the Task object
            # itself, not a task id.
            task.step(advance)
            return self

        def add_task(self, description, total, *args, **kwargs):
            # [FUNCTION DOC] Creates a Task with the given description and total, appends it to self.tasks and
            # returns it (rich returns
            # an id instead). Extra args/kwargs are ignored.
            task = Task(description, total)
            self.tasks.append(task)
            return task
