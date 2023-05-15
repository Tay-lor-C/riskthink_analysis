from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
	'start_date': datetime(2023,5,10)
	'owner': 'airflow'
}

dag = DAG('my_dag', default_args = default_args, schedual_interval = None)

# Python Operator to execute process.py
process_task = PythonOperator(
	task_id = 'process',
	python_callable = ' ~/desktop/code/python/riskthink_analysis/venv/process.py',
	dag=dag
	)

# Python Operator to execute process.py
features_task = PythonOperator(
	task_id = 'features',
	python_callable = '~/desktop/code/python/riskthink_analysis/venv/features.py',
	dag = dag
	)

run_model_task = PythonOperator(
	task_id = 'run_model',
	python_callable = ' ~/desktop/code/python/riskthink_analysis/venv/run_model.py',
	dag = dag
	)

# Bash operator to launch flask API
api_task = BashOperator(
	task_id = 'launch_api',
	bash_command = 'python3 ~/desktop/code/python/riskthink_analysis/venv/api.py /~/desktop/code/python/riskthink_analysis/venv/rf_model.joblib',
	dag = dag
	)

# Set task dependencies
process_task >> features_task >> run_model_task >> api_task





