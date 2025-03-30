from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "v0"
    tasks = {
        "algorithms" : ["v0"],
        "datasets": ["indian_pines"],
        "target_sizes" : [30]
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=True, verbose=True)
    summary, details = ev.evaluate()
