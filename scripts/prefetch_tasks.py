from lm_eval.tasks import initialize_tasks, get_task_dict
initialize_tasks()
tasks = get_task_dict(['piqa','winogrande','hellaswag','arc_easy','arc_challenge','lambada_openai'])
for name,task in tasks.items():
 print(name, {split:len(data) for split,data in task.dataset.items()}, flush=True)
