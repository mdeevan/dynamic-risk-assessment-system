# R0914:Too many local variables
# W0212: Access to a protected member 
# R0903: Too few public methods

ENV_NAME=dynamic_risk_assessment
# ENV_FILE=requirement.txt
ENV_FILE=./config/environment.yml

create-conda-env:
	conda env create -f $(ENV_FILE) -n $(ENV_NAME)

update-conda-env:
	conda env update -f $(ENV_FILE) -n $(ENV_NAME)

update-env:
	pip install -r config/requirements.txt

test:
	# pytest  -vv census_class_test.py main_test.py
	python -m pytest census_class_test.py main_test.py -vv --cov

format:
	black  --line-length 88 *.py

lint:
	pylint src/*.py --disable=W0212,R0914,R0903  
	pylint *.py
# --ignore-patterns=sanitycheck.py setup.py

	