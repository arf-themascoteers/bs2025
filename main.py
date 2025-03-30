from task_runner import TaskRunner

if __name__ == '__main__':
    tag = "tformer2"
    tasks = {
        "algorithms" : ["tformer2"],
        "datasets": ["indian_pines"],
        "target_sizes" : [15]
    }
    ev = TaskRunner(tasks,tag,skip_all_bands=True, verbose=True)
    summary, details = ev.evaluate()
