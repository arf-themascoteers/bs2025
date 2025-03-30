from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "tformer"
    tasks = {
        "algorithms" : ["tformer1"],
        "datasets": ["indian_pines"],
        "target_sizes" : [30]
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=True, verbose=True)
    summary, details = ev.evaluate()
